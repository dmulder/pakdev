#!/usr/bin/env python3
"""
General Purpose Package Updater Script (pakdev)

Searches OBS and IBS for packages, gathers version information from multiple
sources (packtrack, _service files, upstream), and assists with package updates
using either traditional OBS workflow or src-git workflow.

Usage:
  python pkg_update.py <package_name>
  python pkg_update.py himmelblau --ai-provider claude
  python pkg_update.py --help
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from html.parser import HTMLParser

# API endpoints
OBS_API = "https://api.opensuse.org"
IBS_API = "https://api.suse.de"
PACKTRACK_URL = "https://packtrack.suse.cz"

# src-git endpoints
SRC_OPENSUSE = "src.opensuse.org"
SRC_SUSE = "src.suse.de"

TIMEOUT_SECONDS = 60


def print_color(text: str, color: str):
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def run_cmd(
    cmd: list[str],
    cwd: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command with timeout and error handling."""
    try:
        if capture:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
            )
        else:
            result = subprocess.run(cmd, cwd=cwd, timeout=timeout, check=check)
        return result
    except subprocess.TimeoutExpired:
        print_color(f"Command timed out: {' '.join(cmd)}", "red")
        raise
    except subprocess.CalledProcessError:
        # Don't print error here, let caller handle it
        raise


def normalize_version(version: str) -> str:
    """
    Normalize a version string for comparison.

    Examples:
        v2.3.1 -> 2.3.1
        2.3.1+git0.2418ec2 -> 2.3.1
        v2.3.1+git0.2418ec2 -> 2.3.1
        1.4.2+git.0.52da279 -> 1.4.2
        2.3.1~rc1 -> 2.3.1~rc1 (keep pre-release markers)
        2.3.1-beta -> 2.3.1-beta (keep pre-release markers)
    """
    if not version:
        return ""

    # Remove leading 'v' or 'V'
    normalized = re.sub(r'^[vV]', '', version.strip())

    # Remove git suffix patterns: +git0.hash, +git.0.hash, .git0.hash
    normalized = re.sub(r'[+.]git\.?\d*\.?[a-f0-9]+$', '', normalized, flags=re.IGNORECASE)

    # Remove OBS revision suffixes like .1, .2 at the very end after version
    # But be careful not to remove legitimate version parts like 2.3.1
    # Only remove if it looks like an OBS build number (single digit after dash or tilde section)

    return normalized


def versions_match(version1: str, version2: str) -> bool:
    """
    Check if two version strings refer to the same base version.

    Examples:
        versions_match("2.3.1", "v2.3.1+git0.abc123") -> True
        versions_match("1.4.2", "1.4.2+git.0.52da279") -> True
        versions_match("2.3.1", "2.3.2") -> False
    """
    return normalize_version(version1) == normalize_version(version2)


@dataclass
class PackageInstance:
    """Represents a package instance in OBS or IBS."""

    server: str  # "obs" or "ibs"
    api_url: str  # Full API URL
    project: str
    package: str
    version: Optional[str] = None
    release: Optional[str] = None
    src_git_url: Optional[str] = None  # If package is git-managed
    src_git_branch: Optional[str] = None

    @property
    def is_git_managed(self) -> bool:
        return self.src_git_url is not None

    def __str__(self) -> str:
        git_info = (
            f" [git: {self.src_git_url}#{self.src_git_branch}]"
            if self.is_git_managed
            else ""
        )
        ver_info = f" v{self.version}" if self.version else ""
        return f"{self.server}:{self.project}/{self.package}{ver_info}{git_info}"


@dataclass
class CodestreamInfo:
    """Information about a package in a specific codestream from packtrack."""

    codestream: str  # e.g., "openSUSE:Factory", "SUSE:SLE-15-SP7:Update"
    version: str
    server: str  # "obs" or "ibs"
    link: Optional[str] = None  # Link to the package in that codestream
    bugs: Optional[int] = None


@dataclass
class UpstreamVersion:
    """Information about an upstream version."""

    version: str
    url: Optional[str] = None
    release_date: Optional[str] = None
    is_prerelease: bool = False


@dataclass
class PackageInfo:
    """Aggregated information about a package from all sources."""

    name: str
    instances: list[PackageInstance] = field(default_factory=list)
    codestreams: list[CodestreamInfo] = field(default_factory=list)
    upstream_versions: list[UpstreamVersion] = field(default_factory=list)
    upstream_url: Optional[str] = None  # From _service or spec file
    upstream_type: Optional[str] = None  # "github", "gitlab", "other"
    upstream_version_packtrack: Optional[str] = None  # From release-monitoring.org
    devel_project: Optional[str] = None
    maintainers: list[str] = field(default_factory=list)


class PacktrackHTMLParser(HTMLParser):
    """Parse packtrack HTML page to extract codestream information."""

    def __init__(self):
        super().__init__()
        self.codestreams: list[CodestreamInfo] = []
        self.upstream_version: Optional[str] = None
        self._in_codestream_row = False
        self._current_codestream: Optional[str] = None
        self._current_server: Optional[str] = None
        self._current_link: Optional[str] = None
        self._in_version_td = False
        self._current_version: Optional[str] = None
        self._capture_upstream = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == "tr" and "class" in attrs_dict:
            if "codestream" in attrs_dict["class"]:
                self._in_codestream_row = True
                self._current_codestream = None
                self._current_server = None
                self._current_link = None
                self._current_version = None

        elif tag == "a" and self._in_codestream_row:
            href = attrs_dict.get("href", "")
            if "/package/show/" in href:
                self._current_link = href
                # Determine server from URL
                if "build.suse.de" in href:
                    self._current_server = "ibs"
                else:
                    self._current_server = "obs"

        elif tag == "td" and self._in_codestream_row:
            css_class = attrs_dict.get("class", "")
            if "centered" in css_class and not self._current_version:
                self._in_version_td = True

        elif tag == "span":
            title = attrs_dict.get("title", "")
            if title and self._capture_upstream:
                # This might be the upstream version span
                pass

    def handle_endtag(self, tag):
        if tag == "tr" and self._in_codestream_row:
            # Save the codestream if we have all the info
            if self._current_codestream and self._current_version:
                self.codestreams.append(
                    CodestreamInfo(
                        codestream=self._current_codestream,
                        version=self._current_version,
                        server=self._current_server or "obs",
                        link=self._current_link,
                    )
                )
            self._in_codestream_row = False

        elif tag == "td":
            self._in_version_td = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return

        if self._in_codestream_row:
            # Check if this is a codestream name (project:something pattern)
            if ":" in data and not data.startswith("http"):
                # Could be codestream name like "openSUSE:Factory" or "SUSE:SLE-15-SP7:Update"
                if any(
                    x in data
                    for x in ["openSUSE:", "SUSE:", "home:", "network:", "devel:"]
                ):
                    self._current_codestream = data.strip()

            elif self._in_version_td:
                # This is the version
                self._current_version = data.strip()

        # Check for upstream version
        if "upstream" in data.lower():
            self._capture_upstream = True
            # Try to extract version from "upstream X.Y.Z" pattern
            match = re.search(r"upstream\s+([0-9][0-9a-zA-Z._-]*)", data, re.I)
            if match:
                self.upstream_version = match.group(1)
            self._capture_upstream = False


class PackageSearcher:
    """Search for packages in OBS and IBS."""

    def __init__(self):
        self.obs_available = self._check_osc_access(OBS_API)
        self.ibs_available = self._check_osc_access(IBS_API)

    def _check_osc_access(self, api_url: str) -> bool:
        """Check if we have osc access to the given API."""
        try:
            result = run_cmd(
                ["osc", "-A", api_url, "whois"],
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def search_obs(self, package_name: str) -> list[PackageInstance]:
        """Search OBS for packages matching the name."""
        if not self.obs_available:
            print_color("  OBS not available (not logged in?)", "yellow")
            return []

        return self._search_api(OBS_API, "obs", package_name)

    def search_ibs(self, package_name: str) -> list[PackageInstance]:
        """Search IBS for packages matching the name."""
        if not self.ibs_available:
            print_color("  IBS not available (VPN? not logged in?)", "yellow")
            return []

        return self._search_api(IBS_API, "ibs", package_name)

    def _search_api(
        self, api_url: str, server: str, package_name: str
    ) -> list[PackageInstance]:
        """Search for packages using osc search."""
        instances = []

        try:
            # Search for package name matches
            result = run_cmd(
                ["osc", "-A", api_url, "search", "-s", package_name],
                timeout=TIMEOUT_SECONDS,
                check=False,
            )

            if result.returncode != 0:
                return instances

            # Parse results
            # Format:
            # ####################################################################
            # matches for 'himmelblau' in projects:
            # # Project
            # home:scabrero:himmelblau
            # ####################################################################
            # matches for 'himmelblau' in packages:
            # # Project                          # Package
            # SUSE:SLE-15-SP7:GA                 himmelblau

            in_packages_section = False
            for line in result.stdout.strip().split("\n"):
                line = line.strip()

                if "matches for" in line and "in packages:" in line:
                    in_packages_section = True
                    continue

                if line.startswith("#") or not line:
                    continue

                if in_packages_section:
                    # Parse "Project  Package" format
                    parts = line.split()
                    if len(parts) >= 2:
                        project = parts[0]
                        pkg = parts[1]

                        # Only include exact matches
                        if pkg == package_name:
                            instance = PackageInstance(
                                server=server,
                                api_url=api_url,
                                project=project,
                                package=pkg,
                            )
                            instances.append(instance)

        except Exception as e:
            print_color(f"  Search error on {server}: {e}", "yellow")

        return instances

    def get_package_meta(self, instance: PackageInstance) -> Optional[str]:
        """Get package metadata to check for scmsync."""
        try:
            result = run_cmd(
                [
                    "osc",
                    "-A",
                    instance.api_url,
                    "meta",
                    "pkg",
                    instance.project,
                    instance.package,
                ],
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None

    def detect_git_workflow(self, instance: PackageInstance) -> None:
        """Check if a package uses src-git workflow and update instance."""
        meta = self.get_package_meta(instance)
        if meta:
            # Check for scmsync tag
            # Format: https://src.suse.de/pool/pkg?trackingbranch=slfo-main#<commit_hash>
            match = re.search(r"<scmsync>([^<]+)</scmsync>", meta)
            if match:
                scmsync_url = match.group(1)
                # Parse the URL to extract components
                # Remove the commit hash fragment first
                url_without_fragment = scmsync_url.split("#")[0]
                parsed = urllib.parse.urlparse(url_without_fragment)

                # Base URL without query params
                instance.src_git_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                # Extract trackingbranch from query params
                query_params = urllib.parse.parse_qs(parsed.query)
                if "trackingbranch" in query_params:
                    instance.src_git_branch = query_params["trackingbranch"][0]


class PacktrackClient:
    """Client for packtrack.suse.cz (screen scraping)."""

    def get_package_info(
        self, server: str, project: str, package: str
    ) -> tuple[list[CodestreamInfo], Optional[str]]:
        """Scrape packtrack HTML page for package info."""
        url = f"{PACKTRACK_URL}/package/{server}/{project}/{package}/"

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "pakdev/1.0"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                html = response.read().decode()

            parser = PacktrackHTMLParser()
            parser.feed(html)

            return parser.codestreams, parser.upstream_version

        except Exception as e:
            print_color(f"  Packtrack scrape failed: {e}", "yellow")
            return [], None


class ServiceFileParser:
    """Parse OBS _service files to extract source information."""

    def __init__(self, content: str):
        self.content = content
        self.url: Optional[str] = None
        self.scm: Optional[str] = None
        self.revision: Optional[str] = None
        self.version_format: Optional[str] = None
        self.version_rewrite_pattern: Optional[str] = None
        self._parse()

    def _parse(self):
        """Parse the _service XML content."""
        # Extract URL
        url_match = re.search(r'<param name="url">([^<]+)</param>', self.content)
        if url_match:
            self.url = url_match.group(1)

        # Extract SCM type
        scm_match = re.search(r'<param name="scm">([^<]+)</param>', self.content)
        if scm_match:
            self.scm = scm_match.group(1)

        # Extract revision/branch
        rev_match = re.search(r'<param name="revision">([^<]+)</param>', self.content)
        if rev_match:
            self.revision = rev_match.group(1)

        # Extract versionformat (for tar_scm)
        vf_match = re.search(r'<param name="versionformat">([^<]+)</param>', self.content)
        if vf_match:
            self.version_format = vf_match.group(1)

        # Extract versionrewrite-pattern (indicates tag naming convention)
        vrp_match = re.search(r'<param name="versionrewrite-pattern">([^<]+)</param>', self.content)
        if vrp_match:
            self.version_rewrite_pattern = vrp_match.group(1)


class SpecFileParser:
    """Parse RPM spec files to extract source information."""

    def __init__(self, content: str):
        self.content = content
        self.url: Optional[str] = None
        self.version: Optional[str] = None
        self.name: Optional[str] = None
        self._parse()

    def _parse(self):
        """Parse spec file content."""
        for line in self.content.split("\n"):
            line = line.strip()

            if line.lower().startswith("url:"):
                self.url = line.split(":", 1)[1].strip()
            elif line.lower().startswith("version:"):
                self.version = line.split(":", 1)[1].strip()
            elif line.lower().startswith("name:"):
                self.name = line.split(":", 1)[1].strip()


class UpstreamVersionDetector:
    """Detect upstream versions using various methods including AI."""

    def __init__(self, ai_provider: str = "claude", ai_cli_path: Optional[str] = None):
        self.ai_provider = ai_provider
        self.ai_cli_path = ai_cli_path or ai_provider

    def _is_ai_available(self) -> bool:
        """Check if AI CLI is available."""
        return shutil.which(self.ai_cli_path) is not None

    def _detect_source_type(self, url: str) -> str:
        """Detect the type of source repository."""
        if "github.com" in url:
            return "github"
        elif "gitlab" in url:
            return "gitlab"
        elif "git" in url:
            return "git"
        else:
            return "other"

    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pakdev/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                return response.read().decode()
        except Exception:
            return None

    def _extract_github_owner_repo(self, url: str) -> Optional[tuple[str, str]]:
        """Extract owner and repo from GitHub URL."""
        patterns = [
            r"github\.com[/:]([^/]+)/([^/\.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), match.group(2).replace(".git", "")
        return None

    def detect_github_versions(self, url: str) -> list[UpstreamVersion]:
        """Detect versions from GitHub releases/tags."""
        versions = []

        owner_repo = self._extract_github_owner_repo(url)
        if not owner_repo:
            return versions

        owner, repo = owner_repo

        # Try GitHub API for releases
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        content = self._fetch_url(api_url)

        if content:
            try:
                releases = json.loads(content)
                for release in releases[:20]:  # Limit to recent 20
                    tag = release.get("tag_name", "")
                    # Clean version string
                    version = re.sub(r"^[vV]", "", tag)
                    versions.append(
                        UpstreamVersion(
                            version=version,
                            url=release.get("html_url"),
                            release_date=release.get("published_at", "")[:10],
                            is_prerelease=release.get("prerelease", False),
                        )
                    )
            except json.JSONDecodeError:
                pass

        # Also try tags API if no releases found
        if not versions:
            tags_url = f"https://api.github.com/repos/{owner}/{repo}/tags"
            content = self._fetch_url(tags_url)
            if content:
                try:
                    tags = json.loads(content)
                    for tag in tags[:20]:
                        tag_name = tag.get("name", "")
                        version = re.sub(r"^[vV]", "", tag_name)
                        versions.append(
                            UpstreamVersion(
                                version=version,
                                url=f"https://github.com/{owner}/{repo}/releases/tag/{tag_name}",
                            )
                        )
                except json.JSONDecodeError:
                    pass

        return versions

    def detect_with_ai(
        self, url: str, current_versions: list[str]
    ) -> list[UpstreamVersion]:
        """Use AI to detect upstream versions by analyzing web pages."""
        if not self._is_ai_available():
            print_color(f"  AI CLI ({self.ai_provider}) not available", "yellow")
            return []

        source_type = self._detect_source_type(url)

        # Build prompt for AI
        prompt = f"""Analyze this source URL and find available upstream versions.

Source URL: {url}
Source Type: {source_type}
Currently packaged versions: {', '.join(current_versions) if current_versions else 'unknown'}

Instructions:
1. If this is a GitHub/GitLab URL, check the releases and tags pages
2. Look for version numbers in standard formats (e.g., 1.2.3, v1.2.3)
3. Identify which versions are newer than the currently packaged versions
4. Note any pre-release or beta versions

Output your findings as a JSON array with this format:
[
  {{"version": "2.3.1", "url": "https://...", "is_prerelease": false}},
  ...
]

Only output the JSON array, no other text."""

        try:
            result = run_cmd(
                [self.ai_cli_path, "--print", prompt],
                timeout=120,
                check=False,
            )

            if result.returncode == 0 and result.stdout:
                # Try to parse JSON from output
                output = result.stdout.strip()
                # Find JSON array in output
                match = re.search(r"\[[\s\S]*\]", output)
                if match:
                    data = json.loads(match.group())
                    return [
                        UpstreamVersion(
                            version=v.get("version", ""),
                            url=v.get("url"),
                            is_prerelease=v.get("is_prerelease", False),
                        )
                        for v in data
                        if v.get("version")
                    ]
        except Exception as e:
            print_color(f"  AI version detection failed: {e}", "yellow")

        return []

    def detect_versions(
        self, url: str, current_versions: list[str]
    ) -> list[UpstreamVersion]:
        """Detect upstream versions using all available methods."""
        versions = []

        source_type = self._detect_source_type(url)

        # Try source-specific detection first
        if source_type == "github":
            versions = self.detect_github_versions(url)

        # If no versions found or for other sources, try AI
        if not versions:
            print_color("  Using AI to detect upstream versions...", "blue")
            versions = self.detect_with_ai(url, current_versions)

        # Sort versions (try semantic versioning)
        def version_key(v: UpstreamVersion) -> tuple:
            parts = re.split(r"[.\-_]", v.version)
            result = []
            for p in parts[:4]:
                try:
                    result.append(int(re.sub(r"[^0-9]", "", p) or "0"))
                except ValueError:
                    result.append(0)
            while len(result) < 4:
                result.append(0)
            return tuple(result)

        try:
            versions.sort(key=version_key, reverse=True)
        except Exception:
            pass

        return versions


def find_git_tag_for_version(git_url: str, version: str) -> Optional[str]:
    """Find the actual git tag name for a given version.

    Checks common tag naming patterns: version, vVERSION, VERSION, release-VERSION, etc.
    Returns the exact tag name if found, None otherwise.
    """
    if not git_url or not version:
        return None

    try:
        result = run_cmd(
            ["git", "ls-remote", "--tags", git_url],
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return None

        # Parse tags from output
        tags = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                tag = parts[1].replace("refs/tags/", "").rstrip("^{}")
                tags.append(tag)

        # Remove duplicates (some tags have ^{} suffix for annotated tags)
        tags = list(set(tags))

        # Check various tag naming patterns
        candidates = [
            version,                # 0.3.16
            f"v{version}",          # v0.3.16
            f"V{version}",          # V0.3.16
            f"release-{version}",   # release-0.3.16
            f"release_{version}",   # release_0.3.16
            f"{version}-release",   # 0.3.16-release
        ]

        for candidate in candidates:
            if candidate in tags:
                return candidate

        # Also check case-insensitive match
        tags_lower = {t.lower(): t for t in tags}
        for candidate in candidates:
            if candidate.lower() in tags_lower:
                return tags_lower[candidate.lower()]

        return None

    except Exception:
        return None


class ReleaseNoteFetcher:
    """Fetch release notes from various upstream sources."""

    def __init__(self, ai_provider: str = "claude"):
        self.ai_provider = ai_provider

    def _fetch_url(self, url: str, headers: Optional[dict] = None) -> Optional[str]:
        """Fetch URL content."""
        try:
            default_headers = {"User-Agent": "pakdev/1.0"}
            if headers:
                default_headers.update(headers)
            req = urllib.request.Request(url, headers=default_headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                return response.read().decode()
        except Exception:
            return None

    def _extract_github_owner_repo(self, url: str) -> Optional[tuple[str, str]]:
        """Extract owner and repo from GitHub URL."""
        patterns = [
            r"github\.com[/:]([^/]+)/([^/\.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), match.group(2).replace(".git", "")
        return None

    def _extract_gitlab_info(self, url: str) -> Optional[tuple[str, str, str]]:
        """Extract host, owner, and repo from GitLab URL."""
        # Handle various GitLab URL formats
        patterns = [
            r"(gitlab\.[^/]+)[/:]([^/]+)/([^/\.]+)",
            r"(git\.[^/]+)[/:]([^/]+)/([^/\.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1), match.group(2), match.group(3).replace(".git", "")
        return None

    def fetch_github_release_notes(self, url: str, version: str) -> Optional[str]:
        """Fetch release notes from GitHub for a specific version."""
        owner_repo = self._extract_github_owner_repo(url)
        if not owner_repo:
            return None

        owner, repo = owner_repo

        # Try to find the release by tag (with and without 'v' prefix)
        tag_variants = [version, f"v{version}", f"V{version}"]

        for tag in tag_variants:
            api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
            content = self._fetch_url(api_url)

            if content:
                try:
                    release = json.loads(content)
                    body = release.get("body", "")
                    if body:
                        return self._format_release_notes(body, release.get("name", f"Version {version}"))
                except json.JSONDecodeError:
                    pass

        # If no specific release, try to get release list and find matching version
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        content = self._fetch_url(api_url)

        if content:
            try:
                releases = json.loads(content)
                for release in releases:
                    tag = release.get("tag_name", "")
                    # Normalize and compare versions
                    tag_version = re.sub(r"^[vV]", "", tag)
                    if tag_version == version or tag_version == re.sub(r"^[vV]", "", version):
                        body = release.get("body", "")
                        if body:
                            return self._format_release_notes(body, release.get("name", f"Version {version}"))
            except json.JSONDecodeError:
                pass

        return None

    def fetch_gitlab_release_notes(self, url: str, version: str) -> Optional[str]:
        """Fetch release notes from GitLab for a specific version."""
        gitlab_info = self._extract_gitlab_info(url)
        if not gitlab_info:
            return None

        host, owner, repo = gitlab_info

        # URL-encode the project path
        project_path = urllib.parse.quote(f"{owner}/{repo}", safe="")

        # Try to find the release by tag
        tag_variants = [version, f"v{version}", f"V{version}"]

        for tag in tag_variants:
            encoded_tag = urllib.parse.quote(tag, safe="")
            api_url = f"https://{host}/api/v4/projects/{project_path}/releases/{encoded_tag}"
            content = self._fetch_url(api_url)

            if content:
                try:
                    release = json.loads(content)
                    description = release.get("description", "")
                    if description:
                        return self._format_release_notes(description, release.get("name", f"Version {version}"))
                except json.JSONDecodeError:
                    pass

        return None

    def fetch_changelog_from_repo(self, git_url: str, version: str) -> Optional[str]:
        """Try to fetch CHANGELOG file from repository and extract relevant section."""
        # Convert URL to raw file URL for common hosts
        if "github.com" in git_url:
            owner_repo = self._extract_github_owner_repo(git_url)
            if owner_repo:
                owner, repo = owner_repo
                # Try common changelog file names
                changelog_files = [
                    "CHANGELOG.md", "CHANGELOG.rst", "CHANGELOG.txt", "CHANGELOG",
                    "CHANGES.md", "CHANGES.rst", "CHANGES.txt", "CHANGES",
                    "NEWS.md", "NEWS.rst", "NEWS.txt", "NEWS",
                    "HISTORY.md", "HISTORY.rst", "HISTORY.txt", "HISTORY",
                ]

                for filename in changelog_files:
                    # Try main/master branches
                    for branch in ["main", "master"]:
                        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
                        content = self._fetch_url(raw_url)
                        if content:
                            section = self._extract_changelog_section(content, version)
                            if section:
                                return section

        return None

    def _extract_changelog_section(self, content: str, version: str) -> Optional[str]:
        """Extract the section for a specific version from a changelog file."""
        lines = content.split("\n")
        section_lines = []
        in_section = False
        version_patterns = [
            re.escape(version),
            re.escape(f"v{version}"),
            re.escape(f"V{version}"),
        ]

        version_regex = "|".join(version_patterns)

        for line in lines:
            # Check if this line starts a version section
            # Common patterns: ## 1.2.3, ## [1.2.3], # Version 1.2.3, 1.2.3 (date), etc.
            if re.search(rf"^#{{1,3}}\s*\[?({version_regex})\]?", line) or \
               re.search(rf"^({version_regex})\s*[\(\[]", line) or \
               re.search(rf"^Version\s+({version_regex})", line, re.IGNORECASE):
                in_section = True
                section_lines.append(line)
                continue

            if in_section:
                # Check if we've hit the next version section
                if re.match(r"^#{{1,3}}\s*\[?\d+\.\d+", line) or \
                   re.match(r"^\d+\.\d+\.\d+\s*[\(\[]", line) or \
                   re.match(r"^Version\s+\d+\.\d+", line, re.IGNORECASE):
                    break
                section_lines.append(line)

        if section_lines:
            return "\n".join(section_lines).strip()

        return None

    def _format_release_notes(self, body: str, title: str = "") -> str:
        """Format release notes for use in RPM changelog."""
        # Clean up markdown formatting for RPM changelog
        lines = body.strip().split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Convert markdown headers to plain text
            line = re.sub(r"^#{1,6}\s+", "", line)

            # Convert markdown links to plain text
            line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)

            # Keep bullet points but normalize them
            if line.startswith(("- ", "* ", "+ ")):
                line = "  * " + line[2:]
            elif line.startswith(("- ", "* ", "+ ")) is False and formatted_lines:
                # Continuation of previous bullet
                if formatted_lines[-1].startswith("  * "):
                    line = "    " + line

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def fetch_with_ai(self, package_name: str, url: str, version: str, from_version: Optional[str] = None) -> Optional[str]:
        """Use AI to research and generate release notes."""
        if not shutil.which(self.ai_provider):
            return None

        print_color(f"  Using AI to research changes for {package_name} {version}...", "blue")

        version_range = f"from {from_version} to {version}" if from_version else f"for version {version}"

        prompt = f"""Research the changes in {package_name} {version_range}.

Package: {package_name}
URL: {url}
Target Version: {version}
{f'Previous Version: {from_version}' if from_version else ''}

Instructions:
1. Look up the release notes, changelog, or commit history for this package
2. Summarize the key changes (new features, bug fixes, breaking changes)
3. Format your response as bullet points suitable for an RPM changelog
4. Focus on user-visible changes, not internal refactoring
5. Keep it concise but informative (aim for 3-10 bullet points)
6. If there are security fixes, mention them prominently

Output format (just the bullet points, no introduction):
  * First change description
  * Second change description
  * etc.

If you cannot find specific release notes, indicate that clearly."""

        try:
            result = run_cmd(
                [self.ai_provider, "--print", prompt],
                timeout=120,
                check=False,
            )

            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                # Clean up the output - remove any preamble text
                lines = output.split("\n")
                bullet_lines = [l for l in lines if l.strip().startswith(("*", "-", "•"))]
                if bullet_lines:
                    # Normalize bullet format
                    formatted = []
                    for line in bullet_lines:
                        line = line.strip()
                        line = re.sub(r"^[\*\-\•]\s*", "  * ", line)
                        formatted.append(line)
                    return "\n".join(formatted)
                # If no bullet points found, return the full output but clean it up
                return output

        except Exception as e:
            print_color(f"  AI changelog generation failed: {e}", "yellow")

        return None

    def _cleanup_with_ai(self, raw_notes: str, package_name: str, version: str) -> str:
        """Use AI to clean up and format release notes for RPM changelog."""
        if not shutil.which(self.ai_provider):
            return raw_notes

        prompt = f"""Clean up these release notes for an RPM changelog entry.

Package: {package_name}
Version: {version}

Raw release notes:
{raw_notes}

Instructions:
1. Convert to concise bullet points starting with "- " (dash space)
2. Remove GitHub/GitLab specific formatting (PR numbers, contributor mentions, "What's Changed", "New Contributors", "Full Changelog" links)
3. Focus on USER-VISIBLE changes: new features, bug fixes, security fixes, breaking changes
4. Remove internal/CI changes unless they affect users
5. Keep each bullet point to one line, be concise but informative
6. If there are security fixes, list them first
7. Remove any URLs or links
8. Maximum 10 bullet points, prioritize the most important changes

Output format (just the bullet points, nothing else):
- First change
- Second change
- etc.

Output ONLY the bullet points, no introduction or explanation."""

        try:
            result = run_cmd(
                [self.ai_provider, "--print", prompt],
                timeout=60,
                check=False,
            )

            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                # Extract only lines that start with "-" or "*"
                lines = output.split("\n")
                bullet_lines = [l.strip() for l in lines if l.strip().startswith(("-", "*"))]
                if bullet_lines:
                    # Normalize to "- " format
                    formatted = []
                    for line in bullet_lines:
                        line = re.sub(r"^[\*\-]\s*", "- ", line)
                        formatted.append(line)
                    return "\n".join(formatted)

        except Exception:
            pass

        return raw_notes

    def fetch_release_notes(
        self,
        package_name: str,
        url: str,
        version: str,
        from_version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Fetch release notes from various sources in order of preference.

        Returns formatted release notes or None if nothing found.
        """
        print_color(f"  Fetching release notes for {package_name} {version}...", "blue")

        raw_notes = None
        source = None

        # 1. Try GitHub releases
        if "github.com" in url:
            raw_notes = self.fetch_github_release_notes(url, version)
            if raw_notes:
                source = "GitHub"

        # 2. Try GitLab releases
        if not raw_notes and ("gitlab" in url or "git." in url):
            raw_notes = self.fetch_gitlab_release_notes(url, version)
            if raw_notes:
                source = "GitLab"

        # 3. Try CHANGELOG file in repo
        if not raw_notes:
            raw_notes = self.fetch_changelog_from_repo(url, version)
            if raw_notes:
                source = "CHANGELOG file"

        # If we found notes, clean them up with AI
        if raw_notes and source:
            print_color(f"    Found {source} release notes", "green")
            print_color(f"    Formatting with AI...", "blue")
            cleaned_notes = self._cleanup_with_ai(raw_notes, package_name, version)
            if cleaned_notes != raw_notes:
                print_color(f"    Release notes formatted", "green")
            return cleaned_notes

        # 4. Fall back to AI to research from scratch
        print_color("    No release notes found, asking AI to research...", "yellow")
        notes = self.fetch_with_ai(package_name, url, version, from_version)
        if notes:
            print_color("    AI generated changelog entries", "green")
            return notes

        return None


class PackageUpdater:
    """Handles the actual package update process."""

    TIMEOUT_SERVICE = 300  # 5 minutes for osc service
    TIMEOUT_BUILD = 1800  # 30 minutes for builds

    def __init__(
        self,
        instance: PackageInstance,
        target_version: str,
        ai_provider: str = "claude",
    ):
        self.instance = instance
        self.target_version = target_version
        self.ai_provider = ai_provider
        self.work_dir: Optional[Path] = None
        self.branch_project: Optional[str] = None
        self.release_note_fetcher = ReleaseNoteFetcher(ai_provider)
        self.is_git_workflow = False
        self.service_parser: Optional[ServiceFileParser] = None

    def _confirm(self, prompt: str, default_yes: bool = False) -> bool:
        """Ask user for confirmation."""
        try:
            hint = "[Y/n]" if default_yes else "[y/N]"
            response = input(f"\n{prompt} {hint}: ").strip().lower()
            if not response:
                return default_yes
            return response in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print()
            return False

    def _prompt_input(self, prompt: str, default: str = "") -> str:
        """Prompt user for input with optional default."""
        try:
            if default:
                result = input(f"{prompt} [{default}]: ").strip()
                return result if result else default
            else:
                return input(f"{prompt}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return default

    def _update_service_file(self, new_revision: str) -> bool:
        """Update the _service file with new revision/version."""
        if not self.work_dir:
            return False

        service_file = self.work_dir / "_service"
        if not service_file.exists():
            print_color("  No _service file found", "yellow")
            return True  # Not an error, some packages don't use _service

        try:
            content = service_file.read_text()
            original_content = content

            # Update revision parameter
            if '<param name="revision">' in content:
                content = re.sub(
                    r'(<param name="revision">)[^<]*(</param>)',
                    rf'\g<1>{new_revision}\g<2>',
                    content,
                )

            # Also check for version parameter (some services use this)
            if '<param name="version">' in content and self.target_version:
                content = re.sub(
                    r'(<param name="version">)[^<]*(</param>)',
                    rf'\g<1>{self.target_version}\g<2>',
                    content,
                )

            if content != original_content:
                service_file.write_text(content)
                print_color(f"  Updated _service file with revision: {new_revision}", "green")
            else:
                print_color("  No changes needed in _service file", "blue")

            return True

        except Exception as e:
            print_color(f"  Failed to update _service file: {e}", "red")
            return False

    def _clean_cached_service_files(self) -> None:
        """Clean up cached service files that may interfere with fresh fetches.

        Removes:
        - _servicedata: Contains cached revision info that may cause stale results
        - Old tarballs: May prevent fetching new versions
        """
        if not self.work_dir:
            return

        cleaned = []

        # Remove _servicedata if it exists
        servicedata = self.work_dir / "_servicedata"
        if servicedata.exists():
            try:
                servicedata.unlink()
                cleaned.append("_servicedata")
            except Exception as e:
                print_color(f"  Warning: Could not remove _servicedata: {e}", "yellow")

        # Remove old versioned tarballs if we're updating to a specific version
        # Only remove tarballs that appear to be versioned source archives (contain version-like patterns)
        # Preserve auxiliary tarballs like vendor.tar.zst, cargo_config.tar.gz, etc.
        if self.target_version:
            # Pattern to detect versioned tarballs: name-X.Y.Z.tar.* or name_X.Y.Z.tar.*
            version_pattern = re.compile(r'[-_]\d+\.\d+')

            for tarball in self.work_dir.glob("*.tar.*"):
                # Only consider tarballs that have a version number in their name
                if version_pattern.search(tarball.name):
                    # Check if it's an old version tarball (doesn't have target version)
                    if self.target_version not in tarball.name:
                        try:
                            tarball.unlink()
                            cleaned.append(tarball.name)
                        except Exception as e:
                            print_color(f"  Warning: Could not remove {tarball.name}: {e}", "yellow")

            # Also check for .tgz files
            for tarball in self.work_dir.glob("*.tgz"):
                if version_pattern.search(tarball.name):
                    if self.target_version not in tarball.name:
                        try:
                            tarball.unlink()
                            cleaned.append(tarball.name)
                        except Exception as e:
                            print_color(f"  Warning: Could not remove {tarball.name}: {e}", "yellow")

        if cleaned:
            print_color(f"  Cleaned cached files: {', '.join(cleaned)}", "blue")

    def _detect_service_mode(self) -> str:
        """Detect whether _service uses disabled or manual mode.

        Returns 'disabled', 'manual', or 'default' based on the primary service mode.
        """
        if not self.work_dir:
            return "default"

        service_file = self.work_dir / "_service"
        if not service_file.exists():
            return "default"

        try:
            content = service_file.read_text()
            # Check for mode="disabled" on key services like tar_scm
            if 'mode="disabled"' in content:
                return "disabled"
            elif 'mode="manual"' in content:
                return "manual"
            else:
                return "default"
        except Exception:
            return "default"

    def _run_osc_service(self) -> bool:
        """Run osc service to fetch/update sources.

        Automatically detects whether to use 'mr', 'disabledrun', or 'manualrun'
        based on the _service file configuration.
        """
        if not self.work_dir:
            return False

        # Clean up cached files that might interfere
        print_color("\nPreparing to run source services...", "blue")
        self._clean_cached_service_files()

        # Detect which service command to use
        service_mode = self._detect_service_mode()

        if service_mode == "disabled":
            cmd = ["osc", "service", "disabledrun"]
            print_color("  Detected mode='disabled', using 'osc service disabledrun'", "blue")
        elif service_mode == "manual":
            cmd = ["osc", "service", "manualrun"]
            print_color("  Detected mode='manual', using 'osc service manualrun'", "blue")
        else:
            cmd = ["osc", "service", "mr"]
            print_color("  Using 'osc service mr'", "blue")

        print_color("\nRunning source services (this may take a while)...", "blue")

        try:
            # Run service with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output
            for line in process.stdout:
                print(f"  {line}", end="")

            process.wait(timeout=self.TIMEOUT_SERVICE)

            if process.returncode != 0:
                print_color(f"  {' '.join(cmd)} returned non-zero (may be ok)", "yellow")

            # Handle file changes (osc addremove)
            run_cmd(["osc", "addremove"], cwd=self.work_dir, timeout=60, check=False)

            print_color("  Service completed", "green")
            return True

        except subprocess.TimeoutExpired:
            process.kill()
            print_color(f"  {' '.join(cmd)} timed out", "red")
            return False
        except Exception as e:
            print_color(f"  {' '.join(cmd)} failed: {e}", "red")
            return False

    def _verify_version_after_service(self) -> tuple[bool, Optional[str]]:
        """Verify the version after osc service mr matches expected version.

        Returns (version_matches, detected_version).
        """
        if not self.work_dir or not self.target_version:
            return True, None  # Can't verify, assume OK

        detected_version = None

        # Check spec file for Version:
        spec_files = list(self.work_dir.glob("*.spec"))
        if spec_files:
            try:
                content = spec_files[0].read_text()
                parser = SpecFileParser(content)
                if parser.version:
                    detected_version = parser.version
            except Exception:
                pass

        # Also check for tarball names
        if not detected_version:
            tarballs = list(self.work_dir.glob("*.tar.*")) + list(self.work_dir.glob("*.tgz"))
            for tarball in tarballs:
                # Try to extract version from tarball name
                # Common patterns: package-1.2.3.tar.gz, package_1.2.3.tar.bz2
                match = re.search(r'[-_](\d+\.\d+(?:\.\d+)*)', tarball.name)
                if match:
                    detected_version = match.group(1)
                    break

        if detected_version:
            # Normalize versions for comparison (remove leading v, etc)
            norm_target = re.sub(r'^[vV]', '', self.target_version)
            norm_detected = re.sub(r'^[vV]', '', detected_version)

            if norm_detected == norm_target:
                print_color(f"  Version verified: {detected_version}", "green")
                return True, detected_version
            else:
                print_color(f"  Version mismatch! Expected: {self.target_version}, Got: {detected_version}", "red")
                return False, detected_version

        print_color("  Could not detect version from spec/tarball", "yellow")
        return True, None  # Can't verify

    def _handle_spec_only_version_fix(self, detected_version: Optional[str]) -> bool:
        """Handle version mismatch for packages without _service files.

        These packages typically need manual tarball download and spec file updates.
        Returns True if changes were made and we should re-verify.
        """
        if not self.work_dir:
            return False

        print_color("  This package has no _service file.", "blue")
        print_color("  Manual update required: download new tarball and update spec file.", "blue")

        # Find spec file
        spec_files = list(self.work_dir.glob("*.spec"))
        if not spec_files:
            print_color("  No spec file found!", "red")
            return False

        spec_file = spec_files[0]
        spec_content = spec_file.read_text()
        spec_parser = SpecFileParser(spec_content)

        print_color(f"\n  Current spec file: {spec_file.name}", "blue")
        print_color(f"  Current version in spec: {spec_parser.version}", "blue")
        if spec_parser.url:
            print_color(f"  Upstream URL: {spec_parser.url}", "blue")

        # Show options
        print_color("\n  Options:", "yellow")
        print("    1. Get AI help to update the package")
        print("    2. Open a shell to fix manually")
        print("    3. Continue anyway (version mismatch)")
        print("    4. Abort")

        try:
            choice = input("  Choice [1/2/3/4]: ").strip()
            if choice == "1":
                if self._run_ai_spec_update(spec_parser, detected_version):
                    return True
                # If AI didn't fix it, ask again
                return self._handle_spec_only_version_fix(detected_version)
            elif choice == "2":
                self._open_workspace_shell()
                if self._confirm("Did you update the package? Re-verify version?"):
                    return True
                return self._handle_spec_only_version_fix(detected_version)
            elif choice == "3":
                return False  # Continue without fixing
            else:
                raise KeyboardInterrupt
        except (KeyboardInterrupt, EOFError):
            print_color("  Aborted", "red")
            raise

        return False

    def _run_ai_spec_update(self, spec_parser: SpecFileParser, detected_version: Optional[str]) -> bool:
        """Launch interactive AI session to help update a spec-file-only package.

        Returns True if the AI session resulted in changes that should be verified.
        """
        # Check if AI CLI is available
        ai_cli = shutil.which(self.ai_provider)
        if not ai_cli:
            fallback = "gemini" if self.ai_provider == "claude" else "claude"
            ai_cli = shutil.which(fallback)
            if ai_cli:
                self.ai_provider = fallback

        if not ai_cli:
            print_color("  No AI CLI available (claude or gemini)", "red")
            return False

        # Get spec file content
        spec_content = ""
        if self.work_dir:
            spec_files = list(self.work_dir.glob("*.spec"))
            if spec_files:
                try:
                    spec_content = spec_files[0].read_text()
                except Exception:
                    pass

        # List files in workspace
        file_list = ""
        if self.work_dir:
            try:
                files = [f.name for f in self.work_dir.iterdir() if f.is_file()]
                file_list = "\n".join(files)
            except Exception:
                pass

        prompt = f"""I need to update an RPM package from version {detected_version} to version {self.target_version}.

This package does NOT have a _service file, so it requires manual updates:
1. Download the new source tarball for version {self.target_version}
2. Update the Version: line in the spec file
3. Update the Source0: URL if needed
4. Remove the old tarball

Package: {self.instance.package}
Current version: {detected_version}
Target version: {self.target_version}
Upstream URL: {spec_parser.url or 'unknown'}
Workspace: {self.work_dir}

Files in workspace:
{file_list}

Here's the spec file:
```
{spec_content[:6000] if spec_content else '(could not read spec file)'}
```

Please help me update this package to version {self.target_version}. You can:
1. Download the new tarball using curl or wget
2. Edit the spec file to update the Version and Source0 lines
3. Remove old tarballs
4. Run 'osc addremove' to track file changes

IMPORTANT: When you have completed the update or when the user indicates they are done,
please tell them to type 'exit' or press Ctrl+D to exit this AI session and return to
the package updater script, where they can verify the changes.
"""

        print_color("\n  Launching interactive AI session for spec update...", "blue")
        print_color("  (Type 'exit' or press Ctrl+D when done to return to the updater)\n", "yellow")

        try:
            subprocess.run(
                [self.ai_provider, prompt],
                cwd=self.work_dir,
            )
        except Exception as e:
            print_color(f"  AI session failed: {e}", "red")
            return False

        print_color("\n  AI session ended.", "blue")
        return self._confirm("Re-verify the version?", default_yes=True)

    def _diagnose_and_fix_version(self, detected_version: Optional[str]) -> bool:
        """Try to diagnose and fix version mismatch issues.

        Returns True if the issue was fixed and service should be re-run.
        """
        if not self.work_dir:
            return False

        service_file = self.work_dir / "_service"
        has_service_file = service_file.exists()

        parser = None
        if has_service_file:
            content = service_file.read_text()
            parser = ServiceFileParser(content)

        print_color("\n  Diagnosing version mismatch...", "yellow")

        # If no _service file, this is a spec-file-only package
        if not has_service_file:
            return self._handle_spec_only_version_fix(detected_version)

        # Check if it's a git tag naming issue
        if parser.scm == "git" and parser.url:
            current_revision = parser.revision

            # If current revision doesn't look like a proper tag, try to find the right one
            if current_revision:
                print_color(f"  Current revision in _service: {current_revision}", "blue")

                # Check if this revision exists as a tag
                actual_tag = find_git_tag_for_version(parser.url, self.target_version)

                if actual_tag and actual_tag != current_revision:
                    print_color(f"  Found correct tag: {actual_tag} (was using: {current_revision})", "green")

                    if self._confirm(f"Update revision to '{actual_tag}' and re-run service?", default_yes=True):
                        self._update_service_file(actual_tag)
                        return True
                else:
                    print_color(f"  Could not find a valid tag for version {self.target_version}", "yellow")

                    # List available tags that might match
                    try:
                        result = run_cmd(
                            ["git", "ls-remote", "--tags", parser.url],
                            timeout=60,
                            check=False,
                        )
                        if result.returncode == 0:
                            # Find tags containing the version number
                            matching_tags = []
                            for line in result.stdout.strip().split("\n"):
                                if self.target_version in line:
                                    parts = line.split("\t")
                                    if len(parts) >= 2:
                                        tag = parts[1].replace("refs/tags/", "").rstrip("^{}")
                                        matching_tags.append(tag)

                            if matching_tags:
                                print_color(f"  Tags containing '{self.target_version}':", "blue")
                                for tag in set(matching_tags)[:5]:
                                    print(f"    - {tag}")

                                new_tag = self._prompt_input("  Enter correct tag name", "")
                                if new_tag:
                                    self._update_service_file(new_tag)
                                    return True
                    except Exception:
                        pass

        # If we can't auto-fix, ask user what to do
        print_color("\n  Options:", "yellow")
        print("    1. Enter a different revision/tag manually")
        print("    2. Get AI help to diagnose the issue")
        print("    3. Continue anyway (version mismatch)")
        print("    4. Abort")

        try:
            choice = input("  Choice [1/2/3/4]: ").strip()
            if choice == "1":
                new_rev = self._prompt_input("  Enter new revision", "")
                if new_rev:
                    self._update_service_file(new_rev)
                    return True
            elif choice == "2":
                # Launch interactive AI session to help diagnose
                if self._run_ai_diagnosis(parser, detected_version):
                    return True  # AI helped fix it, retry
                # If AI session didn't fix it, fall through to ask again
                return self._diagnose_and_fix_version(detected_version)
            elif choice == "3":
                return False  # Continue without fixing
            else:
                raise KeyboardInterrupt  # Abort
        except (KeyboardInterrupt, EOFError):
            print_color("  Aborted", "red")
            raise

        return False

    def _run_ai_diagnosis(self, parser: ServiceFileParser, detected_version: Optional[str]) -> bool:
        """Launch interactive AI session to help diagnose version mismatch.

        Returns True if the AI session resulted in a fix that should be retried.
        """
        # Check if AI CLI is available
        ai_cli = shutil.which(self.ai_provider)
        if not ai_cli:
            # Try fallback
            fallback = "gemini" if self.ai_provider == "claude" else "claude"
            ai_cli = shutil.which(fallback)
            if ai_cli:
                self.ai_provider = fallback

        if not ai_cli:
            print_color("  No AI CLI available (claude or gemini)", "red")
            return False

        # Build context for AI
        service_content = ""
        if self.work_dir:
            service_file = self.work_dir / "_service"
            if service_file.exists():
                service_content = service_file.read_text()

        spec_content = ""
        if self.work_dir:
            spec_files = list(self.work_dir.glob("*.spec"))
            if spec_files:
                # Just get the first ~50 lines for context
                try:
                    lines = spec_files[0].read_text().split("\n")[:50]
                    spec_content = "\n".join(lines)
                except Exception:
                    pass

        prompt = f"""I'm trying to update a package to version {self.target_version}, but after running 'osc service mr',
the resulting version is {detected_version} instead.

Here's the _service file:
```xml
{service_content}
```

Here's the beginning of the spec file:
```
{spec_content}
```

The upstream git URL is: {parser.url or 'unknown'}
The current revision in _service is: {parser.revision or 'unknown'}

Please help me diagnose why the wrong version was fetched and what the correct revision/tag should be.

IMPORTANT: When you have identified the fix or when the user indicates they are done,
please tell them to type 'exit' or press Ctrl+D to exit this AI session and return to
the package updater script, where they can apply the fix and retry.
"""

        print_color("\n  Launching interactive AI session for diagnosis...", "blue")
        print_color("  (Type 'exit' or press Ctrl+D when done to return to the updater)\n", "yellow")

        try:
            # Run AI CLI interactively with initial prompt
            # Pass prompt as positional argument (not -p which is --print for non-interactive)
            subprocess.run(
                [self.ai_provider, prompt],
                cwd=self.work_dir,
            )
        except Exception as e:
            print_color(f"  AI session failed: {e}", "red")
            return False

        print_color("\n  AI session ended.", "blue")

        # Ask if the user found a fix
        if self._confirm("Did the AI help identify the correct revision/tag?"):
            new_rev = self._prompt_input("  Enter the correct revision/tag", "")
            if new_rev:
                self._update_service_file(new_rev)
                return True

        return False

    def _get_upstream_url(self) -> Optional[str]:
        """Get the upstream URL from service file or spec file."""
        # First try service file
        if self.service_parser and self.service_parser.url:
            return self.service_parser.url

        # Try to find spec file and extract URL
        if self.work_dir:
            spec_files = list(self.work_dir.glob("*.spec"))
            for spec_file in spec_files:
                try:
                    content = spec_file.read_text()
                    parser = SpecFileParser(content)
                    if parser.url:
                        return parser.url
                except Exception:
                    pass

        return None

    def _get_current_version(self) -> Optional[str]:
        """Get the current package version from spec file."""
        if self.work_dir:
            spec_files = list(self.work_dir.glob("*.spec"))
            for spec_file in spec_files:
                try:
                    content = spec_file.read_text()
                    parser = SpecFileParser(content)
                    if parser.version:
                        return parser.version
                except Exception:
                    pass
        return None

    def _create_changelog(self, message: Optional[str] = None) -> bool:
        """Create a changelog entry using osc vc with detailed release notes."""
        if not self.work_dir:
            return False

        # Get upstream URL for fetching release notes
        upstream_url = self._get_upstream_url()
        current_version = self._get_current_version()

        # Find the .changes file
        changes_files = list(self.work_dir.glob("*.changes"))
        changes_file = changes_files[0] if changes_files else self.work_dir / f"{self.instance.package}.changes"

        # Build the changelog message
        if message:
            # User provided a custom message, use it as-is
            final_message = message
        elif self.target_version:
            # Try to fetch release notes automatically first
            release_notes = None
            if upstream_url:
                release_notes = self.release_note_fetcher.fetch_release_notes(
                    package_name=self.instance.package,
                    url=upstream_url,
                    version=self.target_version,
                    from_version=current_version,
                )

            if release_notes:
                # Build changelog with release notes
                final_message = f"Update to version {self.target_version}:\n{release_notes}"
            else:
                # Fallback to simple message
                final_message = f"Update to version {self.target_version}"
        else:
            final_message = "Update to latest version"

        # Show the proposed changelog to user
        print_color("\n" + "=" * 60, "cyan")
        print_color("Proposed changelog entry:", "bold")
        print_color("=" * 60, "cyan")
        print(final_message)
        print_color("=" * 60, "cyan")

        # Allow user to edit or accept
        print_color("\nOptions:", "yellow")
        print("  [Enter] Accept this changelog")
        print("  [e] Edit changelog message manually")
        print("  [a] Use interactive AI to generate changelog (recommended for major updates)")
        print("  [s] Use simple message (just 'Update to version X.Y.Z')")
        print("  [c] Cancel changelog creation")

        try:
            choice = input("\nChoice [Enter/e/a/s/c]: ").strip().lower()

            if choice == "c":
                print_color("  Changelog creation cancelled", "yellow")
                return True  # Don't fail the whole process

            if choice == "s":
                final_message = f"Update to version {self.target_version}" if self.target_version else "Update to latest version"

            elif choice == "e":
                # Let user edit in their editor
                print_color("\nEnter your changelog message (end with a single '.' on a line):", "blue")
                lines = []
                while True:
                    line = input()
                    if line.strip() == ".":
                        break
                    lines.append(line)
                if lines:
                    final_message = "\n".join(lines)

            elif choice == "a":
                # Interactive AI changelog generation
                if self._run_interactive_ai_changelog(changes_file, upstream_url, current_version):
                    print_color("  Changelog updated by AI", "green")
                    return True
                else:
                    print_color("  AI changelog generation cancelled, using proposed entry", "yellow")

            print_color(f"\nCreating changelog entry...", "blue")

            # osc vc with -m for non-interactive mode
            run_cmd(
                ["osc", "vc", "-m", final_message],
                cwd=self.work_dir,
                timeout=30,
            )
            print_color("  Changelog entry created", "green")
            return True
        except KeyboardInterrupt:
            print_color("\n  Changelog creation skipped", "yellow")
            return True
        except Exception as e:
            print_color(f"  Failed to create changelog: {e}", "yellow")
            print_color("  You may need to run 'osc vc' manually", "yellow")
            return True  # Don't fail the whole process

    def _run_interactive_ai_changelog(
        self,
        changes_file: Path,
        upstream_url: Optional[str],
        from_version: Optional[str],
    ) -> bool:
        """Run interactive AI session to generate changelog entry."""
        if not shutil.which(self.ai_provider):
            print_color(f"  AI CLI ({self.ai_provider}) not available", "red")
            return False

        # Get current timestamp in RPM changelog format
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%a %b %d %H:%M:%S UTC %Y")

        # Try to get maintainer info
        try:
            result = run_cmd(["git", "config", "user.name"], timeout=5, check=False)
            maintainer_name = result.stdout.strip() if result.returncode == 0 else "Unknown"
            result = run_cmd(["git", "config", "user.email"], timeout=5, check=False)
            maintainer_email = result.stdout.strip() if result.returncode == 0 else "unknown@example.com"
        except Exception:
            maintainer_name = os.environ.get("USER", "Unknown")
            maintainer_email = "unknown@example.com"

        version_info = ""
        if from_version and self.target_version:
            version_info = f"Previous version in this codestream: {from_version}\nTarget version: {self.target_version}"
        elif self.target_version:
            version_info = f"Target version: {self.target_version}"

        prompt = f"""I need you to create an RPM changelog entry for updating {self.instance.package}.

{version_info}
Package upstream URL: {upstream_url or 'Unknown'}
Changes file: {changes_file}

Instructions:
1. Search the GitHub/GitLab releases page for ALL versions between {from_version or 'the previous version'} and {self.target_version}
2. Compile a comprehensive changelog covering all changes across these versions
3. Filter OUT changes that only affect other distros (Ubuntu, Fedora, Debian fixes) - this is for openSUSE/SLE
4. Focus on user-visible changes: new features, bug fixes, security fixes, breaking changes
5. Format as bullet points starting with "- "

The changelog entry format MUST be:
-------------------------------------------------------------------
{timestamp} - {maintainer_name} <{maintainer_email}>

- Update to version {self.target_version}:
- First change description
- Second change description
- etc.

Once you have gathered the information and composed the changelog entry, use your Edit tool to add it to the TOP of the file: {changes_file}

The new entry should be inserted at the very beginning of the file, BEFORE any existing entries.

When you are done editing the file, tell the user to type 'exit' or '/exit' to continue with the package update process."""

        print_color("\n" + "=" * 60, "cyan")
        print_color("Starting interactive AI session for changelog generation", "bold")
        print_color("=" * 60, "cyan")
        print_color(f"\nThe AI will research releases and edit: {changes_file}", "blue")
        print_color("You can guide the AI if needed (e.g., clarify version ranges, filtering)", "blue")
        print_color("\nType 'exit' or '/exit' when the AI has finished editing the changelog.\n", "yellow")

        try:
            # Run AI interactively
            subprocess.run(
                [self.ai_provider, prompt],
                cwd=self.work_dir,
            )
        except KeyboardInterrupt:
            print_color("\n  AI session cancelled", "yellow")
            return False
        except Exception as e:
            print_color(f"  AI session failed: {e}", "red")
            return False

        # Verify the changes file was modified
        if changes_file.exists():
            print_color("\n  AI session ended.", "blue")
            return self._confirm("Accept the AI-generated changelog?", default_yes=True)

        return False

    def _get_available_repositories(self) -> list[str]:
        """Get list of available repositories for the project."""
        try:
            cmd = [
                "osc", "-A", self.instance.api_url, "repos",
                self.instance.project, self.instance.package
            ]
            result = run_cmd(cmd, timeout=30, check=False)
            if result.returncode == 0 and result.stdout:
                # Parse output - each line is "repo arch" or just "repo"
                repos = set()
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if parts:
                            repos.add(parts[0])
                return sorted(repos)
        except Exception as e:
            print_color(f"  Warning: Could not get repository list: {e}", "yellow")
        return []

    def _select_build_repository(self) -> Optional[str]:
        """Select the best repository for test build, or let user choose.

        Returns the selected repository name, or None to skip the build.
        """
        repos = self._get_available_repositories()

        if not repos:
            print_color("  Warning: Could not determine available repositories.", "yellow")
            # Fall back to defaults based on server type
            if self.instance.server == "ibs":
                return "standard"
            else:
                return "openSUSE_Tumbleweed"

        # Auto-select based on priority preferences
        # For IBS/internal builds: prefer standard
        # For OBS: prefer openSUSE_Tumbleweed, then openSUSE_Factory, then Leap
        preferred_repos = []
        if self.instance.server == "ibs":
            preferred_repos = ["standard", "leap_16", "leap_15"]
        else:
            preferred_repos = ["openSUSE_Tumbleweed", "openSUSE_Factory", "Leap_15.6", "Leap_15.5"]

        # Try to find a preferred repo
        for pref in preferred_repos:
            for repo in repos:
                if repo.lower() == pref.lower() or repo.lower().startswith(pref.lower()):
                    return repo

        # No preferred repo found - ask user
        print_color(f"\n  Available repositories for {self.instance.project}:", "blue")
        for i, repo in enumerate(repos, 1):
            print(f"    {i}. {repo}")
        print(f"    {len(repos) + 1}. Skip test build")

        try:
            while True:
                choice = input(f"  Select repository [1-{len(repos) + 1}]: ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(repos):
                        return repos[idx - 1]
                    elif idx == len(repos) + 1:
                        return None  # Skip build
                # Also accept repo name directly
                if choice in repos:
                    return choice
                print_color(f"  Invalid selection. Enter 1-{len(repos) + 1} or repository name.", "yellow")
        except (KeyboardInterrupt, EOFError):
            return None

    def _run_test_build(self, repo: Optional[str] = None, arch: str = "x86_64") -> tuple[bool, str]:
        """Run a local test build.

        Returns (success, build_log) tuple.
        """
        if not self.work_dir:
            return False, ""

        # Select repository if not provided
        if repo is None:
            repo = self._select_build_repository()
            if repo is None:
                print_color("  Build skipped by user", "yellow")
                return True, ""  # User chose to skip, not a failure

        print_color(f"\nRunning test build ({repo}/{arch})...", "blue")
        print_color("  This may take a while. Press Ctrl+C to skip.", "yellow")

        # Create a temp file for capturing build output
        log_file = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, prefix='pakdev_build_')
        log_path = log_file.name
        log_file.close()

        try:
            # Always specify API URL to ensure correct server is used
            cmd = ["osc", "-A", self.instance.api_url, "build", "--no-verify", repo, arch]

            # For git workflow, use alternative project
            if self.is_git_workflow:
                cmd = ["osc", "-A", self.instance.api_url, "build", "--no-verify",
                       f"--alternative-project={self.instance.project}", repo, arch]

            # Use 'script' to run interactively while capturing output
            # This allows password prompts to work while still logging
            # -q: quiet mode (no script started/done messages)
            # -e: return exit status of the child process
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            script_cmd = ["script", "-q", "-e", "-c", cmd_str, log_path]

            result = subprocess.run(
                script_cmd,
                cwd=self.work_dir,
                timeout=self.TIMEOUT_BUILD,
            )

            # Read the captured log
            build_log = ""
            try:
                with open(log_path, 'r', errors='replace') as f:
                    build_log = f.read()
            except Exception:
                pass

            if result.returncode == 0:
                print_color("\n  Build succeeded!", "green")
                return True, build_log
            else:
                print_color("\n  Build failed", "red")
                return False, build_log

        except subprocess.TimeoutExpired:
            print_color("\n  Build timed out", "red")
            build_log = ""
            try:
                with open(log_path, 'r', errors='replace') as f:
                    build_log = f.read()
            except Exception:
                pass
            return False, build_log
        except KeyboardInterrupt:
            print_color("\n  Build skipped by user", "yellow")
            return True, ""  # User chose to skip, not a failure
        except Exception as e:
            print_color(f"\n  Build error: {e}", "red")
            return False, str(e)
        finally:
            # Clean up temp file
            try:
                os.unlink(log_path)
            except Exception:
                pass

    def _handle_build_failure(self, build_log: str) -> bool:
        """Handle a build failure with options including AI diagnosis.

        Returns True if the build should be retried.
        """
        print_color("\n  Build failure options:", "yellow")
        print("    1. Get AI help to diagnose and fix the build")
        print("    2. Open a shell in the workspace to fix manually")
        print("    3. Continue anyway (skip build)")
        print("    4. Abort")

        try:
            choice = input("  Choice [1/2/3/4]: ").strip()
            if choice == "1":
                if self._run_ai_build_diagnosis(build_log):
                    return True  # Retry the build
                # If AI didn't fix it, ask again
                return self._handle_build_failure(build_log)
            elif choice == "2":
                self._open_workspace_shell()
                if self._confirm("Did you fix the issue? Retry build?"):
                    return True
                return self._handle_build_failure(build_log)
            elif choice == "3":
                return False  # Continue without successful build
            else:
                raise KeyboardInterrupt
        except (KeyboardInterrupt, EOFError):
            print_color("  Aborted", "red")
            raise

        return False

    def _open_workspace_shell(self):
        """Open an interactive shell in the workspace directory."""
        if not self.work_dir:
            return

        print_color(f"\n  Opening shell in {self.work_dir}", "blue")
        print_color("  Type 'exit' when done to return to the updater.\n", "yellow")

        try:
            shell = os.environ.get("SHELL", "/bin/bash")
            subprocess.run([shell], cwd=self.work_dir)
        except Exception as e:
            print_color(f"  Failed to open shell: {e}", "red")

    def _run_ai_build_diagnosis(self, build_log: str) -> bool:
        """Launch interactive AI session to help diagnose build failure.

        Returns True if the AI session resulted in a fix that should be retried.
        """
        # Check if AI CLI is available
        ai_cli = shutil.which(self.ai_provider)
        if not ai_cli:
            # Try fallback
            fallback = "gemini" if self.ai_provider == "claude" else "claude"
            ai_cli = shutil.which(fallback)
            if ai_cli:
                self.ai_provider = fallback

        if not ai_cli:
            print_color("  No AI CLI available (claude or gemini)", "red")
            return False

        # Get the last portion of the build log (most relevant for errors)
        log_lines = build_log.split("\n")
        # Get last 200 lines or full log if shorter
        relevant_log = "\n".join(log_lines[-200:]) if len(log_lines) > 200 else build_log

        # Get spec file content
        spec_content = ""
        if self.work_dir:
            spec_files = list(self.work_dir.glob("*.spec"))
            if spec_files:
                try:
                    spec_content = spec_files[0].read_text()
                except Exception:
                    pass

        # List files in workspace
        file_list = ""
        if self.work_dir:
            try:
                files = [f.name for f in self.work_dir.iterdir() if f.is_file()]
                file_list = "\n".join(files)
            except Exception:
                pass

        prompt = f"""I'm trying to build an RPM package but the build failed. Please help me diagnose and fix the issue.

Package: {self.instance.package}
Target version: {self.target_version or 'unknown'}
Workspace: {self.work_dir}

Files in workspace:
{file_list}

Here's the spec file:
```
{spec_content[:8000] if spec_content else '(could not read spec file)'}
```

Here's the end of the build log (last 200 lines):
```
{relevant_log}
```

Please analyze the build failure and help me fix it. You can:
1. Read files in the workspace to understand the issue better
2. Edit files to fix problems (spec file, patches, etc.)
3. Suggest what changes need to be made

IMPORTANT: When you have fixed the issue or when the user indicates they are done,
please tell them to type 'exit' or press Ctrl+D to exit this AI session and return to
the package updater script, where they can retry the build.
"""

        print_color("\n  Launching interactive AI session for build diagnosis...", "blue")
        print_color("  (Type 'exit' or press Ctrl+D when done to return to the updater)\n", "yellow")

        try:
            # Run AI CLI interactively with initial prompt
            # Pass prompt as positional argument (not -p which is --print for non-interactive)
            subprocess.run(
                [self.ai_provider, prompt],
                cwd=self.work_dir,
            )
        except Exception as e:
            print_color(f"  AI session failed: {e}", "red")
            return False

        print_color("\n  AI session ended.", "blue")

        # Ask if the user wants to retry the build
        return self._confirm("Retry the build?")

    def _osc_commit(self, message: str) -> bool:
        """Commit changes using osc commit."""
        if not self.work_dir:
            return False

        print_color(f"\nCommitting: {message}", "blue")

        try:
            run_cmd(
                ["osc", "commit", "-m", message],
                cwd=self.work_dir,
                timeout=120,
                capture=False,
            )
            print_color("  Committed successfully", "green")
            return True
        except Exception as e:
            print_color(f"  Commit failed: {e}", "red")
            return False

    def _osc_submitrequest(self) -> bool:
        """Submit a request to merge changes."""
        if not self.work_dir:
            return False

        print_color("\nSubmitting merge request...", "blue")

        try:
            message = f"Update {self.instance.package}"
            if self.target_version:
                message += f" to version {self.target_version}"

            result = run_cmd(
                ["osc", "submitrequest", "--yes", "-m", message],
                cwd=self.work_dir,
                timeout=120,
            )
            print_color("  Submit request created!", "green")
            if result.stdout:
                print(result.stdout)
            return True
        except Exception as e:
            print_color(f"  Submit request failed: {e}", "red")
            return False

    def _git_commit_and_push(self, message: str) -> bool:
        """Commit and push changes for git workflow."""
        if not self.work_dir:
            return False

        is_internal = SRC_SUSE in (self.instance.src_git_url or "")

        # Stage all changes
        print_color("\nStaging changes...", "blue")
        try:
            run_cmd(["git", "add", "-A"], cwd=self.work_dir, timeout=30)

            # Show what will be committed
            result = run_cmd(["git", "status", "--short"], cwd=self.work_dir, timeout=30)
            if result.stdout:
                print("  Changes to be committed:")
                for line in result.stdout.strip().split("\n"):
                    print(f"    {line}")
        except Exception as e:
            print_color(f"  Failed to stage changes: {e}", "red")
            return False

        # Commit
        print_color(f"\nCommitting: {message}", "blue")
        try:
            run_cmd(
                ["git", "commit", "-m", message],
                cwd=self.work_dir,
                timeout=30,
            )
            print_color("  Committed successfully", "green")
        except Exception as e:
            print_color(f"  Commit failed: {e}", "red")
            return False

        # Push
        if is_internal:
            # Internal src.suse.de - fork-based workflow
            # Push feature branch to fork (origin), then create PR to pool (upstream)
            feature_branch = getattr(self, '_feature_branch', None)
            tracking_branch = getattr(self, '_tracking_branch', 'main')
            git_host = getattr(self, '_git_host', 'src.suse.de')
            package_name = getattr(self, '_package_name', self.instance.package)

            print_color(f"\nPushing feature branch to your fork...", "blue")
            try:
                # Push feature branch to origin (fork)
                run_cmd(
                    ["git", "push", "-u", "origin", feature_branch or "HEAD"],
                    cwd=self.work_dir,
                    timeout=120,
                    capture=False,
                )
                print_color("  Pushed to fork successfully", "green")
            except Exception as e:
                print_color(f"  Push failed: {e}", "red")
                return False

            # Create PR from fork to pool
            print_color(f"\nCreating pull request to pool/{package_name}...", "blue")

            # Get username from stored value
            username = getattr(self, '_git_username', None)
            if not username:
                # Fallback: try to parse from stored fork URL
                fork_url = getattr(self, '_fork_url', '')
                if fork_url and ':' in fork_url:
                    # Parse from gitea@src.suse.de:username/package.git
                    username = fork_url.split(':')[1].split('/')[0]

            # Construct PR URL for Gitea
            pr_url = f"https://{git_host}/pool/{package_name}/compare/{tracking_branch}...{username}:{feature_branch}"
            print_color(f"\n  To create the PR, visit:", "blue")
            print_color(f"  {pr_url}", "cyan")
            print_color(f"\n  Or use the Gitea web interface:", "blue")
            print_color(f"  https://{git_host}/{username}/{package_name}/pulls", "cyan")
            return True
        else:
            # External src.opensuse.org - use agit PR workflow
            branch_name = f"update-{self.target_version}" if self.target_version else "update"
            print_color(f"\nCreating PR via agit to {branch_name}...", "blue")
            try:
                # Push to refs/for/main/<branch> for agit PR
                target_branch = self.instance.src_git_branch or "main"
                run_cmd(
                    ["git", "push", "origin", f"HEAD:refs/for/{target_branch}/{branch_name}"],
                    cwd=self.work_dir,
                    timeout=120,
                    capture=False,
                )
                print_color("  PR created via agit!", "green")
                return True
            except Exception as e:
                print_color(f"  PR creation failed: {e}", "red")
                print_color("  You can manually push or use git-obs pr create", "yellow")
                return False

    def copy_from_existing_instance(
        self,
        source: PackageInstance,
        searcher: "PackageSearcher",
    ) -> bool:
        """
        Copy package files from an existing instance that already has the target version.

        Handles all combinations:
        - OBS source -> OBS destination
        - OBS source -> git destination
        - git source -> OBS destination
        - git source -> git destination
        """
        print_color("\n=== Copying from existing instance ===", "cyan")
        print_color(f"  Source: {source.project}/{source.package}", "blue")
        print_color(f"  Destination: {self.instance.project}/{self.instance.package}", "blue")

        # Detect if source is git-managed
        if source.src_git_url is None:
            searcher.detect_git_workflow(source)

        source_is_git = source.is_git_managed
        dest_is_git = self.instance.is_git_managed

        print_color(f"  Source workflow: {'git' if source_is_git else 'OBS'}", "blue")
        print_color(f"  Destination workflow: {'git' if dest_is_git else 'OBS'}", "blue")

        # Create temp directory for source files
        source_dir = Path(tempfile.mkdtemp(prefix="pkg-source-"))

        try:
            # Step 1: Fetch source files
            print_color("\nStep 1: Fetching source files...", "blue")
            if source_is_git:
                if not self._fetch_from_git_source(source, source_dir):
                    return False
            else:
                if not self._fetch_from_obs_source(source, source_dir):
                    return False

            # List fetched files
            files = list(source_dir.iterdir())
            print_color(f"  Fetched {len(files)} files:", "green")
            for f in files[:10]:
                print(f"    - {f.name}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")

            # Step 2: Copy to destination
            print_color("\nStep 2: Setting up destination...", "blue")
            if dest_is_git:
                return self._copy_to_git_destination(source_dir)
            else:
                return self._copy_to_obs_destination(source_dir)

        except Exception as e:
            print_color(f"  Copy failed: {e}", "red")
            return False
        finally:
            # Clean up source temp dir
            shutil.rmtree(source_dir, ignore_errors=True)

    def _fetch_from_obs_source(self, source: PackageInstance, dest_dir: Path) -> bool:
        """Fetch package files from an OBS instance."""
        try:
            # List files in the package
            result = run_cmd(
                [
                    "osc", "-A", source.api_url, "ls",
                    source.project, source.package,
                ],
                timeout=60,
            )

            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            if not files:
                print_color("  No files found in source package", "red")
                return False

            # Download each file
            for filename in files:
                # Skip certain files that shouldn't be copied
                # - _link, _aggregate: OBS internal files
                # - *.changes: changelog history should not be copied between projects
                if filename in ("_link", "_aggregate") or filename.endswith(".changes"):
                    print(f"    Skipping {filename} (not copied between projects)")
                    continue

                print(f"    Fetching {filename}...")
                try:
                    result = run_cmd(
                        [
                            "osc", "-A", source.api_url, "cat",
                            source.project, source.package, filename,
                        ],
                        timeout=120,
                    )
                    (dest_dir / filename).write_text(result.stdout)
                except Exception as e:
                    # Try binary mode for tarballs etc
                    try:
                        result = subprocess.run(
                            [
                                "osc", "-A", source.api_url, "cat",
                                source.project, source.package, filename,
                            ],
                            capture_output=True,
                            timeout=120,
                        )
                        if result.returncode == 0:
                            (dest_dir / filename).write_bytes(result.stdout)
                        else:
                            print_color(f"    Warning: Could not fetch {filename}", "yellow")
                    except Exception:
                        print_color(f"    Warning: Could not fetch {filename}", "yellow")

            return True
        except Exception as e:
            print_color(f"  Failed to fetch from OBS: {e}", "red")
            return False

    def _fetch_from_git_source(self, source: PackageInstance, dest_dir: Path) -> bool:
        """Fetch package files from a git-managed source."""
        try:
            git_url = source.src_git_url
            git_branch = source.src_git_branch or "main"

            # Convert HTTPS URL to SSH if needed for src.suse.de
            if SRC_SUSE in git_url and git_url.startswith("https://"):
                # Parse and convert: https://src.suse.de/pool/pkg -> gitea@src.suse.de:pool/pkg.git
                parsed = urllib.parse.urlparse(git_url)
                path = parsed.path.strip("/")
                git_url = f"gitea@{parsed.netloc}:{path}.git"

            print(f"    Cloning {git_url} (branch: {git_branch})...")

            # Clone to temp location
            clone_dir = Path(tempfile.mkdtemp(prefix="pkg-git-clone-"))
            try:
                run_cmd(
                    ["git", "clone", "--depth", "1", "-b", git_branch, git_url, str(clone_dir)],
                    timeout=120,
                    capture=False,
                )

                # Copy files to dest_dir, excluding .git and .changes files
                for item in clone_dir.iterdir():
                    # Skip .git directory
                    if item.name == ".git":
                        continue
                    # Skip changelog files - history should not be copied between projects
                    if item.name.endswith(".changes"):
                        print(f"    Skipping {item.name} (not copied between projects)")
                        continue
                    if item.is_file():
                        shutil.copy2(item, dest_dir / item.name)
                    else:
                        shutil.copytree(item, dest_dir / item.name)

                return True
            finally:
                shutil.rmtree(clone_dir, ignore_errors=True)

        except Exception as e:
            print_color(f"  Failed to fetch from git: {e}", "red")
            return False

    def _copy_to_obs_destination(self, source_dir: Path) -> bool:
        """Copy files to an OBS destination (traditional workflow)."""
        # First, ensure we have a working directory (branch the package)
        if not self.work_dir:
            print_color("\nBranching destination package...", "blue")
            # Similar to update_obs_workflow step 1
            try:
                result = run_cmd(
                    [
                        "osc", "-A", self.instance.api_url, "branch",
                        self.instance.project, self.instance.package,
                    ],
                    timeout=120,
                    check=False,
                )

                if result.returncode != 0 and "already exists" not in (result.stderr or ""):
                    print_color(f"  Branch failed: {result.stderr}", "red")
                    return False

                # Parse branch project
                output = (result.stdout or "") + (result.stderr or "")
                for line in output.split("\n"):
                    if "osc co " in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "co" and i + 1 < len(parts):
                                branch_path = parts[i + 1]
                                if "/" in branch_path:
                                    self.branch_project = branch_path.split("/")[0]
                                break

                if not self.branch_project:
                    self.branch_project = f"home:{os.environ.get('USER', 'unknown')}:branches:{self.instance.project}"

            except Exception as e:
                print_color(f"  Branch failed: {e}", "red")
                return False

            # Checkout the branched package
            self.work_dir = Path(tempfile.mkdtemp(prefix="pkg-update-"))
            try:
                run_cmd(
                    [
                        "osc", "-A", self.instance.api_url, "co",
                        self.branch_project, self.instance.package,
                        "-o", str(self.work_dir),
                    ],
                    timeout=120,
                    capture=False,
                )
            except Exception as e:
                print_color(f"  Checkout failed: {e}", "red")
                return False

        # Copy files from source_dir to work_dir
        print_color("\nCopying files to destination...", "blue")
        for item in source_dir.iterdir():
            dest_path = self.work_dir / item.name
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            if item.is_file():
                shutil.copy2(item, dest_path)
            else:
                shutil.copytree(item, dest_path)
            print(f"    Copied {item.name}")

        print_color(f"\n  Files copied to: {self.work_dir}", "green")
        print_color("  You can now review, build, and commit the changes.", "green")
        return True

    def _setup_git_workspace(self) -> bool:
        """Set up the git workspace for src-git workflow (steps 1-4).

        Sets up: fork, clone, sync with upstream, create feature branch.
        Populates: self.work_dir, self._feature_branch, self._tracking_branch, etc.
        Returns True if successful.
        """
        pool_url = self.instance.src_git_url
        if not pool_url:
            print_color("  No src-git URL detected", "red")
            return False

        tracking_branch = self.instance.src_git_branch or "main"

        # Parse the pool URL to construct SSH URLs
        parsed = urllib.parse.urlparse(pool_url)
        git_host = parsed.netloc
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2 or path_parts[0] != "pool":
            print_color(f"  Unexpected pool URL format: {pool_url}", "red")
            return False

        package_name = path_parts[1]
        pool_ssh_url = f"gitea@{git_host}:pool/{package_name}.git"

        print_color(f"\nPool repository: {pool_ssh_url}", "blue")
        print_color(f"Tracking branch: {tracking_branch}", "blue")

        # Get username for the git host
        print_color(f"\nStep 1: Setting up fork...", "blue")
        username = self._get_git_username(git_host)
        if not username:
            print_color("  Username required for src-git workflow", "red")
            return False

        fork_url = f"gitea@{git_host}:{username}/{package_name}.git"
        print_color(f"  Fork URL: {fork_url}", "blue")

        # Check if fork exists
        if self._check_fork_exists(fork_url):
            print_color(f"  Fork exists", "green")
        else:
            print_color(f"  Fork does not exist at {fork_url}", "yellow")
            print_color(f"  Please create a fork via: https://{git_host}/pool/{package_name}", "yellow")
            if not self._confirm("Have you created the fork? Continue?"):
                return False

        # Clone the fork
        print_color("\nStep 2: Cloning your fork...", "blue")
        if not self._confirm("Proceed with cloning?"):
            return False

        self.work_dir = Path(tempfile.mkdtemp(prefix="pkg-update-"))
        pkg_dir = self.work_dir / package_name

        try:
            run_cmd(
                ["git", "clone", fork_url, str(pkg_dir)],
                timeout=120,
                capture=False,
            )
            self.work_dir = pkg_dir
            print_color(f"  Cloned to: {self.work_dir}", "green")
        except Exception as e:
            print_color(f"  Clone failed: {e}", "red")
            return False

        # Add pool as upstream and sync
        print_color("\nStep 3: Syncing with upstream pool...", "blue")
        try:
            run_cmd(
                ["git", "remote", "add", "upstream", pool_ssh_url],
                cwd=self.work_dir,
                timeout=30,
                check=False,
            )
            run_cmd(
                ["git", "fetch", "upstream"],
                cwd=self.work_dir,
                timeout=120,
                capture=False,
            )
            run_cmd(
                ["git", "checkout", tracking_branch],
                cwd=self.work_dir,
                timeout=30,
                check=False,
            )
            run_cmd(
                ["git", "reset", "--hard", f"upstream/{tracking_branch}"],
                cwd=self.work_dir,
                timeout=30,
            )
            print_color(f"  Synced with upstream/{tracking_branch}", "green")
        except Exception as e:
            print_color(f"  Sync failed: {e}", "red")
            if not self._confirm("Continue anyway?"):
                return False

        # Create feature branch
        print_color("\nStep 4: Creating feature branch...", "blue")
        branch_suffix = f"update-{self.target_version}" if self.target_version else "update"
        feature_branch = f"{tracking_branch}-{branch_suffix}"
        feature_branch = self._prompt_input("  Feature branch name", feature_branch)

        try:
            run_cmd(
                ["git", "checkout", "-b", feature_branch],
                cwd=self.work_dir,
                timeout=30,
            )
            print_color(f"  Created branch: {feature_branch}", "green")
        except Exception as e:
            print_color(f"  Failed to create branch: {e}", "red")
            return False

        # Store for later use
        self._feature_branch = feature_branch
        self._tracking_branch = tracking_branch
        self._pool_url = pool_url
        self._pool_ssh_url = pool_ssh_url
        self._fork_url = fork_url
        self._git_host = git_host
        self._package_name = package_name
        self._git_username = username
        self.is_git_workflow = True

        return True

    def _copy_to_git_destination(self, source_dir: Path) -> bool:
        """Copy files to a git destination (src-git workflow)."""
        # Set up git workspace if not already done
        if not self.work_dir:
            print_color("\nSetting up git workspace...", "blue")
            if not self._setup_git_workspace():
                return False

        # Copy files from source_dir to work_dir
        print_color("\nCopying files to git destination...", "blue")
        for item in source_dir.iterdir():
            dest_path = self.work_dir / item.name
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            if item.is_file():
                shutil.copy2(item, dest_path)
            else:
                shutil.copytree(item, dest_path)
            print(f"    Copied {item.name}")

        print_color(f"\n  Files copied to: {self.work_dir}", "green")
        return True

    def _continue_after_copy_obs(self) -> bool:
        """Continue OBS workflow after copying files from another instance."""
        print_color("\n=== Continuing OBS Workflow (after copy) ===", "cyan")
        self.is_git_workflow = False

        if not self.work_dir:
            print_color("  No work directory set up", "red")
            return False

        # Parse service file if present
        service_file = self.work_dir / "_service"
        if service_file.exists():
            content = service_file.read_text()
            self.service_parser = ServiceFileParser(content)

        # Clean up cached service files to ensure fresh state
        self._clean_cached_service_files()

        # Step 1: Verify version matches
        print_color("\nStep 1: Verifying version...", "blue")
        version_ok, detected_version = self._verify_version_after_service()
        if not version_ok and detected_version:
            print_color(f"  Note: Package is at version {detected_version}", "yellow")
            if self.target_version:
                self.target_version = detected_version

        # Step 2: Create/update changelog
        print_color("\nStep 2: Creating changelog...", "blue")
        self._create_changelog()

        # Step 3: Test build (optional)
        print_color("\nStep 3: Test build...", "blue")
        if self._confirm("Run a test build?", default_yes=True):
            build_success, build_log = self._run_test_build()
            while not build_success:
                try:
                    if self._handle_build_failure(build_log):
                        build_success, build_log = self._run_test_build()
                    else:
                        break
                except KeyboardInterrupt:
                    print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                    return False

        # Step 4: Commit and submit
        print_color("\nStep 4: Commit and submit...", "blue")
        if self._confirm("Ready to commit and create submit request?", default_yes=True):
            return self._osc_commit_and_sr()

        print_color(f"\nWorkspace: {self.work_dir}", "green")
        return True

    def _continue_after_copy_git(self) -> bool:
        """Continue git workflow after copying files from another instance."""
        print_color("\n=== Continuing Git Workflow (after copy) ===", "cyan")
        self.is_git_workflow = True

        if not self.work_dir:
            print_color("  No work directory set up", "red")
            return False

        is_internal = SRC_SUSE in (self.instance.src_git_url or "")

        # Parse service file if present
        service_file = self.work_dir / "_service"
        if service_file.exists():
            content = service_file.read_text()
            self.service_parser = ServiceFileParser(content)

        # Clean up cached service files
        self._clean_cached_service_files()

        # Step 1: Verify version matches
        print_color("\nStep 1: Verifying version...", "blue")
        version_ok, detected_version = self._verify_version_after_service()
        if not version_ok and detected_version:
            print_color(f"  Note: Package is at version {detected_version}", "yellow")
            if self.target_version:
                self.target_version = detected_version

        # Step 2: Create changelog
        print_color("\nStep 2: Creating changelog...", "blue")
        self._create_changelog()

        # Step 3: Test build (optional)
        print_color("\nStep 3: Test build...", "blue")
        if self._confirm("Run a test build?", default_yes=True):
            build_success, build_log = self._run_test_build()
            while not build_success:
                try:
                    if self._handle_build_failure(build_log):
                        build_success, build_log = self._run_test_build()
                    else:
                        break
                except KeyboardInterrupt:
                    print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                    return False

        # Step 4: Commit and push/PR
        print_color("\nStep 4: Commit and push...", "blue")
        commit_msg = f"Update to {self.target_version}" if self.target_version else "Update to latest version"
        if is_internal:
            action = "commit and push"
        else:
            action = "commit and create PR"

        if self._confirm(f"Ready to {action}. Proceed?", default_yes=True):
            if not self._git_commit_and_push(commit_msg):
                print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                return False

        print_color(f"\nWorkspace: {self.work_dir}", "green")
        return True

    def update_obs_workflow(self) -> bool:
        """Update package using traditional OBS workflow."""
        print_color("\n=== Traditional OBS Workflow ===", "cyan")
        self.is_git_workflow = False

        # Step 1: Branch the package
        print_color("\nStep 1: Branching package...", "blue")
        print(
            f"  osc -A {self.instance.api_url} branch {self.instance.project} {self.instance.package}"
        )

        if not self._confirm("Proceed with branching?"):
            return False

        try:
            result = run_cmd(
                [
                    "osc",
                    "-A",
                    self.instance.api_url,
                    "branch",
                    self.instance.project,
                    self.instance.package,
                ],
                timeout=120,
                check=False,
            )

            if result.returncode != 0 and "already exists" not in (
                result.stderr or ""
            ):
                print_color(f"  Branch failed: {result.stderr}", "red")
                return False

            # Parse branch project from output
            output = (result.stdout or "") + (result.stderr or "")
            for line in output.split("\n"):
                if "osc co " in line:
                    match = re.search(r"osc co (\S+)", line)
                    if match:
                        self.branch_project = match.group(1).split("/")[0]
                        break

            if not self.branch_project:
                # Construct it
                result = run_cmd(
                    ["osc", "-A", self.instance.api_url, "whois"],
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    username = result.stdout.strip().split(":")[0].strip()
                    self.branch_project = (
                        f"home:{username}:branches:{self.instance.project}"
                    )

            print_color(f"  Branched to: {self.branch_project}", "green")

        except Exception as e:
            print_color(f"  Branch failed: {e}", "red")
            return False

        # Step 2: Checkout
        print_color("\nStep 2: Checking out package...", "blue")
        self.work_dir = Path(tempfile.mkdtemp(prefix="pkg-update-"))

        try:
            run_cmd(
                [
                    "osc",
                    "-A",
                    self.instance.api_url,
                    "checkout",
                    self.branch_project,
                    self.instance.package,
                ],
                cwd=self.work_dir,
                timeout=120,
            )

            # Find checkout directory
            for item in self.work_dir.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        if subitem.name == self.instance.package:
                            self.work_dir = subitem
                            break

            print_color(f"  Checked out to: {self.work_dir}", "green")

        except Exception as e:
            print_color(f"  Checkout failed: {e}", "red")
            return False

        # Step 3: Update _service file
        print_color("\nStep 3: Updating _service file...", "blue")
        service_file = self.work_dir / "_service"
        if service_file.exists():
            content = service_file.read_text()
            self.service_parser = ServiceFileParser(content)
            current_rev = self.service_parser.revision or "unknown"
            print_color(f"  Current revision: {current_rev}", "blue")

            # Ask for new revision - detect correct git tag format
            suggested_rev = current_rev
            if self.target_version and self.service_parser.url:
                # Check if the service uses tar_scm with git
                if self.service_parser.scm == "git":
                    print_color(f"  Detecting git tag format for version {self.target_version}...", "blue")
                    actual_tag = find_git_tag_for_version(self.service_parser.url, self.target_version)
                    if actual_tag:
                        suggested_rev = actual_tag
                        print_color(f"  Found matching tag: {actual_tag}", "green")
                    else:
                        # Couldn't find exact tag, use version but warn
                        suggested_rev = self.target_version
                        print_color(f"  Warning: Could not find tag for {self.target_version}, using version as-is", "yellow")
                        print_color(f"  Common tag formats to try: {self.target_version}, v{self.target_version}", "yellow")
                else:
                    suggested_rev = self.target_version

            new_revision = self._prompt_input(
                "  Enter new revision (tag/branch/commit)",
                suggested_rev
            )

            if new_revision and new_revision != current_rev:
                self._update_service_file(new_revision)
            else:
                print_color("  Keeping current revision", "blue")
        else:
            print_color("  No _service file found", "yellow")

        # Step 4: Run osc service (if _service file exists)
        has_service_file = (self.work_dir / "_service").exists()
        if has_service_file:
            if self._confirm("Run osc service to fetch sources?", default_yes=True):
                if not self._run_osc_service():
                    if not self._confirm("Service failed. Continue anyway?"):
                        return False

        # Step 5: Verify version matches (always do this)
        print_color("\nStep 5: Verifying version...", "blue")
        version_ok, detected_version = self._verify_version_after_service()

        # Loop to fix version issues
        max_retries = 3
        retry_count = 0
        while not version_ok and retry_count < max_retries:
            retry_count += 1
            try:
                should_retry = self._diagnose_and_fix_version(detected_version)
                if should_retry:
                    if has_service_file:
                        print_color(f"\nRetrying osc service (attempt {retry_count + 1})...", "blue")
                        if self._run_osc_service():
                            version_ok, detected_version = self._verify_version_after_service()
                        else:
                            break
                    else:
                        # For spec-only packages, just re-verify
                        version_ok, detected_version = self._verify_version_after_service()
                else:
                    # User chose to continue without fixing
                    break
            except KeyboardInterrupt:
                print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                return False

        if not version_ok:
            print_color(f"\nWarning: Proceeding with version {detected_version} instead of {self.target_version}", "yellow")
            if detected_version:
                # Update target_version to match what we actually got
                self.target_version = detected_version

        # Step 6: Create changelog
        self._create_changelog()

        # Step 7: Test build (optional)
        if self._confirm("Run a test build?", default_yes=True):
            build_success, build_log = self._run_test_build()
            while not build_success:
                try:
                    if self._handle_build_failure(build_log):
                        # User wants to retry
                        build_success, build_log = self._run_test_build()
                    else:
                        # User chose to continue without successful build
                        break
                except KeyboardInterrupt:
                    print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                    return False

        # Step 8: Commit
        commit_msg = f"Update to {self.target_version}" if self.target_version else "Update to latest version"
        if self._confirm(f"Commit changes with message: '{commit_msg}'?", default_yes=True):
            if not self._osc_commit(commit_msg):
                print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                return False

        # Step 9: Submit request
        if self._confirm("Create submit request to upstream?", default_yes=True):
            self._osc_submitrequest()

        print_color(f"\nWorkspace: {self.work_dir}", "green")
        return True

    def _get_git_username(self, git_host: str) -> Optional[str]:
        """Get the username for a git host (for fork path, not authentication)."""
        # Try to get from gitea-specific config first
        try:
            result = run_cmd(
                ["git", "config", "--get", f"gitea.{git_host}.user"],
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

        # Try OS username as default
        default_user = os.environ.get("USER", "")

        # Prompt user - clarify this is for the fork path, not auth
        print_color(f"  Your fork will be at: gitea@{git_host}:<username>/{self.instance.package}.git", "blue")
        username = self._prompt_input(f"  Enter your Gitea username for {git_host}", default_user)

        # Optionally save for future use
        if username:
            try:
                run_cmd(
                    ["git", "config", "--global", f"gitea.{git_host}.user", username],
                    timeout=10,
                    check=False,
                )
            except Exception:
                pass

        return username if username else None

    def _check_fork_exists(self, fork_url: str) -> bool:
        """Check if a git fork exists by attempting to ls-remote."""
        try:
            result = run_cmd(
                ["git", "ls-remote", fork_url],
                timeout=30,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def update_srcgit_workflow(self) -> bool:
        """Update package using src-git workflow."""
        print_color("\n=== src-git Workflow ===", "cyan")

        # Set up git workspace (steps 1-4: fork, clone, sync, create branch)
        if not self._setup_git_workspace():
            return False

        is_internal = SRC_SUSE in (self.instance.src_git_url or "")

        # Step 5: Update _service file
        print_color("\nStep 5: Updating _service file...", "blue")
        service_file = self.work_dir / "_service"
        if service_file.exists():
            content = service_file.read_text()
            self.service_parser = ServiceFileParser(content)
            current_rev = self.service_parser.revision or "unknown"
            print_color(f"  Current revision: {current_rev}", "blue")

            # Ask for new revision - detect correct git tag format
            suggested_rev = current_rev
            if self.target_version and self.service_parser.url:
                # Check if the service uses tar_scm with git
                if self.service_parser.scm == "git":
                    print_color(f"  Detecting git tag format for version {self.target_version}...", "blue")
                    actual_tag = find_git_tag_for_version(self.service_parser.url, self.target_version)
                    if actual_tag:
                        suggested_rev = actual_tag
                        print_color(f"  Found matching tag: {actual_tag}", "green")
                    else:
                        # Couldn't find exact tag, use version but warn
                        suggested_rev = self.target_version
                        print_color(f"  Warning: Could not find tag for {self.target_version}, using version as-is", "yellow")
                        print_color(f"  Common tag formats to try: {self.target_version}, v{self.target_version}", "yellow")
                else:
                    suggested_rev = self.target_version

            new_revision = self._prompt_input(
                "  Enter new revision (tag/branch/commit)",
                suggested_rev
            )

            if new_revision and new_revision != current_rev:
                self._update_service_file(new_revision)
            else:
                print_color("  Keeping current revision", "blue")
        else:
            print_color("  No _service file found", "yellow")

        # Step 6: Run osc service (if _service file exists)
        has_service_file = (self.work_dir / "_service").exists()
        if has_service_file:
            print_color("\nStep 6: Running osc service...", "blue")
            if self._confirm("Run osc service to fetch sources?", default_yes=True):
                if not self._run_osc_service():
                    if not self._confirm("Service failed. Continue anyway?"):
                        return False

        # Step 7: Verify version matches (always do this)
        print_color("\nStep 7: Verifying version...", "blue")
        version_ok, detected_version = self._verify_version_after_service()

        # Loop to fix version issues
        max_retries = 3
        retry_count = 0
        while not version_ok and retry_count < max_retries:
            retry_count += 1
            try:
                should_retry = self._diagnose_and_fix_version(detected_version)
                if should_retry:
                    if has_service_file:
                        print_color(f"\nRetrying osc service (attempt {retry_count + 1})...", "blue")
                        if self._run_osc_service():
                            version_ok, detected_version = self._verify_version_after_service()
                        else:
                            break
                    else:
                        # For spec-only packages, just re-verify
                        version_ok, detected_version = self._verify_version_after_service()
                else:
                    # User chose to continue without fixing
                    break
            except KeyboardInterrupt:
                print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                return False

        if not version_ok:
            print_color(f"\nWarning: Proceeding with version {detected_version} instead of {self.target_version}", "yellow")
            if detected_version:
                # Update target_version to match what we actually got
                self.target_version = detected_version

        # Step 8: Create changelog
        print_color("\nStep 8: Creating changelog...", "blue")
        self._create_changelog()

        # Step 9: Test build (optional)
        print_color("\nStep 9: Test build...", "blue")
        if self._confirm("Run a test build?", default_yes=True):
            build_success, build_log = self._run_test_build()
            while not build_success:
                try:
                    if self._handle_build_failure(build_log):
                        # User wants to retry
                        build_success, build_log = self._run_test_build()
                    else:
                        # User chose to continue without successful build
                        break
                except KeyboardInterrupt:
                    print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                    return False

        # Step 10: Commit and push/PR
        print_color("\nStep 10: Commit and push...", "blue")
        commit_msg = f"Update to {self.target_version}" if self.target_version else "Update to latest version"
        if is_internal:
            action = "commit and push"
        else:
            action = "commit and create PR"

        if self._confirm(f"Ready to {action}. Proceed?", default_yes=True):
            if not self._git_commit_and_push(commit_msg):
                print_color(f"\nWorkspace preserved at: {self.work_dir}", "yellow")
                return False

        print_color(f"\nWorkspace: {self.work_dir}", "green")
        return True


def _fetch_instance_version(instance: PackageInstance) -> None:
    """Fetch the version of a package instance from its spec file."""
    if instance.version:
        return  # Already have version

    # List of spec file names to try
    spec_names = [
        f"{instance.package}.spec",
        # Some packages have different naming conventions
    ]

    for spec_name in spec_names:
        try:
            result = run_cmd(
                [
                    "osc",
                    "-A",
                    instance.api_url,
                    "cat",
                    instance.project,
                    instance.package,
                    spec_name,
                ],
                timeout=30,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                parser = SpecFileParser(result.stdout)
                if parser.version:
                    instance.version = parser.version
                    return
        except Exception:
            pass

    # If standard names didn't work, list files and find any .spec file
    try:
        result = run_cmd(
            [
                "osc",
                "-A",
                instance.api_url,
                "ls",
                instance.project,
                instance.package,
            ],
            timeout=30,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            for filename in result.stdout.strip().split("\n"):
                if filename.endswith(".spec"):
                    try:
                        spec_result = run_cmd(
                            [
                                "osc",
                                "-A",
                                instance.api_url,
                                "cat",
                                instance.project,
                                instance.package,
                                filename,
                            ],
                            timeout=30,
                            check=False,
                        )
                        if spec_result.returncode == 0 and spec_result.stdout:
                            parser = SpecFileParser(spec_result.stdout)
                            if parser.version:
                                instance.version = parser.version
                                return
                    except Exception:
                        pass
    except Exception:
        pass


def gather_package_info(
    package_name: str,
    searcher: PackageSearcher,
    packtrack: PacktrackClient,
    version_detector: UpstreamVersionDetector,
) -> PackageInfo:
    """Gather all available information about a package."""
    info = PackageInfo(name=package_name)

    # Search OBS and IBS
    print_color("\nSearching OBS...", "blue")
    obs_instances = searcher.search_obs(package_name)
    print_color(f"  Found {len(obs_instances)} instance(s)", "green")

    print_color("\nSearching IBS...", "blue")
    ibs_instances = searcher.search_ibs(package_name)
    print_color(f"  Found {len(ibs_instances)} instance(s)", "green")

    info.instances = obs_instances + ibs_instances

    # Detect git workflow and fetch versions for each instance
    print_color("\nDetecting workflows and fetching versions...", "blue")
    for instance in info.instances:
        searcher.detect_git_workflow(instance)
        # Fetch version from spec file
        _fetch_instance_version(instance)

        workflow = "git-managed" if instance.is_git_managed else "traditional OBS"
        version_str = f" v{instance.version}" if instance.version else ""
        print_color(f"  {instance.project}:{version_str} {workflow}", "blue")

    # Try packtrack for the devel project (usually network:idm or similar)
    # Find a good candidate project for packtrack lookup
    # Priority: devel projects (network:, devel:) > home projects > SUSE/openSUSE projects
    packtrack_project = None
    packtrack_priority = 0  # Higher is better

    for instance in info.instances:
        if instance.server == "obs":
            project = instance.project
            priority = 0

            # Devel projects are highest priority
            if project.startswith(("network:", "devel:")):
                priority = 100
            # Home projects of the maintainer might have good info
            elif project.startswith("home:"):
                priority = 10
            # Factory and staging are lower priority
            elif "Factory" in project or "Staging" in project:
                priority = 1
            # SUSE internal projects on OBS - lowest priority for packtrack
            elif project.startswith("SUSE:"):
                priority = 0
            else:
                priority = 50  # Unknown, give it medium priority

            if priority > packtrack_priority:
                packtrack_priority = priority
                packtrack_project = project

    if packtrack_project:
        print_color(f"\nFetching packtrack info for {packtrack_project}...", "blue")
        codestreams, upstream_version = packtrack.get_package_info(
            "obs", packtrack_project, package_name
        )
        if codestreams:
            info.codestreams = codestreams
            print_color(f"  Found {len(codestreams)} codestream(s)", "green")

            # Deduplicate: remove OBS/IBS instances that are covered by packtrack codestreams
            # This includes the codestream project itself and any branches of it
            codestream_projects = {cs.codestream for cs in codestreams}

            def is_covered_by_packtrack(instance: PackageInstance) -> bool:
                """Check if an instance is already covered by packtrack codestreams."""
                project = instance.project
                # Direct match
                if project in codestream_projects:
                    return True
                # Check if it's a branch of a codestream project
                # Branches are typically: home:user:branches:ORIGINAL_PROJECT
                for cs_project in codestream_projects:
                    if f":branches:{cs_project}" in project:
                        return True
                return False

            original_count = len(info.instances)
            info.instances = [inst for inst in info.instances if not is_covered_by_packtrack(inst)]
            removed_count = original_count - len(info.instances)
            if removed_count > 0:
                print_color(f"  Removed {removed_count} duplicate(s) covered by packtrack", "blue")

        if upstream_version:
            info.upstream_version_packtrack = upstream_version
            print_color(f"  Upstream version (release-monitoring.org): {upstream_version}", "green")

    # Get _service file to find upstream URL
    print_color("\nFetching source information...", "blue")
    for instance in info.instances:
        try:
            result = run_cmd(
                [
                    "osc",
                    "-A",
                    instance.api_url,
                    "cat",
                    instance.project,
                    instance.package,
                    "_service",
                ],
                timeout=60,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                parser = ServiceFileParser(result.stdout)
                if parser.url:
                    info.upstream_url = parser.url
                    info.upstream_type = version_detector._detect_source_type(
                        parser.url
                    )
                    print_color(f"  Found upstream URL: {info.upstream_url}", "green")
                    break
        except Exception:
            pass

    # If no _service, try spec file
    if not info.upstream_url:
        for instance in info.instances:
            try:
                result = run_cmd(
                    [
                        "osc",
                        "-A",
                        instance.api_url,
                        "cat",
                        instance.project,
                        instance.package,
                        f"{instance.package}.spec",
                    ],
                    timeout=60,
                    check=False,
                )
                if result.returncode == 0 and result.stdout:
                    parser = SpecFileParser(result.stdout)
                    if parser.url:
                        info.upstream_url = parser.url
                        info.upstream_type = version_detector._detect_source_type(
                            parser.url
                        )
                        print_color(
                            f"  Found upstream URL from spec: {info.upstream_url}",
                            "green",
                        )
                    if parser.version:
                        # Update instance version
                        instance.version = parser.version
                    break
            except Exception:
                pass

    # Detect upstream versions
    if info.upstream_url:
        print_color("\nDetecting upstream versions...", "blue")
        current_versions = [cs.version for cs in info.codestreams if cs.version] + [
            inst.version for inst in info.instances if inst.version
        ]

        info.upstream_versions = version_detector.detect_versions(
            info.upstream_url, current_versions
        )
        if info.upstream_versions:
            print_color(
                f"  Found {len(info.upstream_versions)} upstream version(s)", "green"
            )

    return info


def display_package_info(info: PackageInfo):
    """Display gathered package information to the user."""
    print()
    print_color("=" * 70, "cyan")
    print_color(f"  Package: {info.name}", "bold")
    print_color("=" * 70, "cyan")

    if info.upstream_url:
        print_color(f"\nUpstream: {info.upstream_url}", "blue")

    if info.upstream_version_packtrack:
        print_color(
            f"Latest upstream (release-monitoring.org): {info.upstream_version_packtrack}",
            "green",
        )

    # Display instances found via osc search
    if info.instances:
        print_color("\nPackage Instances (from osc search):", "bold")
        for i, inst in enumerate(info.instances, 1):
            git_marker = " [GIT]" if inst.is_git_managed else ""
            version = f" (v{inst.version})" if inst.version else ""
            print(
                f"  {i}. [{inst.server.upper()}] {inst.project}/{inst.package}{version}{git_marker}"
            )
            if inst.src_git_url:
                print(f"      Git: {inst.src_git_url}")
                if inst.src_git_branch:
                    print(f"      Branch: {inst.src_git_branch}")

    # Display codestreams from packtrack
    if info.codestreams:
        print_color("\nCodestreams (from packtrack):", "bold")
        for cs in info.codestreams:
            server_tag = f"[{cs.server.upper()}]" if cs.server else ""
            bugs_str = f" ({cs.bugs} bugs)" if cs.bugs else ""
            print(f"  - {server_tag} {cs.codestream}: {cs.version}{bugs_str}")

    # Display upstream versions
    if info.upstream_versions:
        print_color("\nUpstream Versions Available:", "bold")
        for i, v in enumerate(info.upstream_versions[:10], 1):
            prerelease = " [pre-release]" if v.is_prerelease else ""
            date = f" ({v.release_date})" if v.release_date else ""
            print(f"  {i}. {v.version}{prerelease}{date}")
        if len(info.upstream_versions) > 10:
            print(f"  ... and {len(info.upstream_versions) - 10} more")

    print()


def find_instances_with_version(
    info: PackageInfo,
    target_version: str,
    exclude_instance: Optional[PackageInstance] = None,
) -> list[tuple[str, PackageInstance | CodestreamInfo]]:
    """
    Find all instances/codestreams that already have the target version.

    Returns list of (source_type, instance_or_codestream) tuples.
    source_type is "instance" or "codestream".
    """
    matches = []

    if not target_version:
        return matches

    norm_target = normalize_version(target_version)

    # Check instances
    for inst in info.instances:
        if inst.version and normalize_version(inst.version) == norm_target:
            # Skip the instance we're updating
            if exclude_instance and inst.project == exclude_instance.project:
                continue
            matches.append(("instance", inst))

    # Check codestreams
    for cs in info.codestreams:
        if cs.version and normalize_version(cs.version) == norm_target:
            # Skip if this codestream matches the excluded instance
            if exclude_instance and cs.codestream == exclude_instance.project:
                continue
            matches.append(("codestream", cs))

    return matches


def select_update_target(
    info: PackageInfo,
    searcher: PackageSearcher,
) -> Optional[tuple[PackageInstance, str]]:
    """Let user select which instance to update and to which version."""
    if not info.instances and not info.codestreams:
        print_color("No package instances or codestreams found to update.", "red")
        return None

    # Build combined list of options
    options: list[tuple[str, PackageInstance | CodestreamInfo]] = []

    # Add instances from osc search
    for inst in info.instances:
        options.append(("instance", inst))

    # Add codestreams from packtrack (that aren't already in instances)
    instance_projects = {inst.project for inst in info.instances}
    for cs in info.codestreams:
        if cs.codestream not in instance_projects:
            options.append(("codestream", cs))

    # Select instance/codestream
    print_color("\nSelect package to update:", "yellow")
    print_color("  -- Package Instances (from osc search) --", "blue")
    idx = 1
    for opt_type, opt in options:
        if opt_type == "instance":
            inst = opt
            git_marker = " [GIT]" if inst.is_git_managed else ""
            ver = f" (v{inst.version})" if inst.version else ""
            print(
                f"  {idx}. [{inst.server.upper()}] {inst.project}/{inst.package}{ver}{git_marker}"
            )
        else:
            # This is a codestream
            if idx == len(info.instances) + 1:
                print_color("  -- Codestreams (from packtrack) --", "blue")
            cs = opt
            print(f"  {idx}. [{cs.server.upper()}] {cs.codestream} (v{cs.version})")
        idx += 1
    print("  0. Cancel")

    try:
        choice = input("\nEnter choice: ").strip()
        if choice == "0" or not choice:
            return None
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(options):
            print_color("Invalid selection", "red")
            return None

        opt_type, selected = options[choice_idx]

        if opt_type == "instance":
            selected_instance = selected
        else:
            # Convert codestream to PackageInstance
            cs = selected
            api_url = IBS_API if cs.server == "ibs" else OBS_API
            selected_instance = PackageInstance(
                server=cs.server,
                api_url=api_url,
                project=cs.codestream,
                package=info.name,
                version=cs.version,
            )
            # Detect git workflow for this instance
            print_color(f"\nDetecting workflow for {cs.codestream}...", "blue")
            searcher.detect_git_workflow(selected_instance)
            if selected_instance.is_git_managed:
                print_color(
                    f"  Git-managed: {selected_instance.src_git_url} (branch: {selected_instance.src_git_branch})",
                    "green",
                )
            else:
                print_color("  Traditional OBS workflow", "blue")

    except (ValueError, KeyboardInterrupt, EOFError):
        return None

    # Select version
    print_color("\nSelect target version:", "yellow")
    if info.upstream_versions:
        for i, v in enumerate(info.upstream_versions[:10], 1):
            prerelease = " [pre-release]" if v.is_prerelease else ""
            print(f"  {i}. {v.version}{prerelease}")
    print("  M. Enter manually")
    print("  E. Empty (let _service select next version)")
    print("  0. Cancel")

    try:
        choice = input("\nEnter choice: ").strip().upper()
        if choice == "0" or not choice:
            return None
        elif choice == "M":
            version = input("Enter version: ").strip()
            if not version:
                return None
        elif choice == "E":
            version = ""
        else:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(info.upstream_versions):
                print_color("Invalid selection", "red")
                return None
            version = info.upstream_versions[idx].version
    except (ValueError, KeyboardInterrupt, EOFError):
        return None

    return selected_instance, version


def offer_copy_from_existing(
    info: PackageInfo,
    target_instance: PackageInstance,
    target_version: str,
    searcher: PackageSearcher,
) -> Optional[PackageInstance]:
    """
    Check if any existing instance has the target version and offer to copy from it.

    Returns the source instance to copy from, or None if user declines or no match found.
    """
    if not target_version:
        return None

    # Find instances with the target version
    matches = find_instances_with_version(info, target_version, target_instance)

    if not matches:
        return None

    # Show matches to user
    print_color("\n" + "=" * 70, "yellow")
    print_color(f"  Found {len(matches)} instance(s) already at version {target_version}!", "yellow")
    print_color("=" * 70, "yellow")

    for i, (match_type, match) in enumerate(matches, 1):
        if match_type == "instance":
            inst = match
            git_marker = " [GIT]" if inst.is_git_managed else ""
            ver = f" (v{inst.version})" if inst.version else ""
            print(f"  {i}. [{inst.server.upper()}] {inst.project}/{inst.package}{ver}{git_marker}")
        else:
            cs = match
            print(f"  {i}. [{cs.server.upper()}] {cs.codestream} (v{cs.version}) [packtrack]")

    print("  0. Skip - do normal update from upstream")

    print_color("\nCopying from an existing instance avoids duplicate work!", "green")

    try:
        choice = input("\nCopy from which instance? [0 to skip]: ").strip()
        if not choice or choice == "0":
            return None

        idx = int(choice) - 1
        if idx < 0 or idx >= len(matches):
            print_color("Invalid selection, skipping copy", "yellow")
            return None

        match_type, selected = matches[idx]

        if match_type == "instance":
            source_instance = selected
        else:
            # Convert codestream to PackageInstance
            cs = selected
            api_url = IBS_API if cs.server == "ibs" else OBS_API
            source_instance = PackageInstance(
                server=cs.server,
                api_url=api_url,
                project=cs.codestream,
                package=info.name,
                version=cs.version,
            )
            # Detect git workflow for source
            searcher.detect_git_workflow(source_instance)

        return source_instance

    except (ValueError, KeyboardInterrupt, EOFError):
        return None


def detect_workspace_info(workspace_path: Path) -> Optional[dict]:
    """Detect information about an existing workspace.

    Returns a dict with:
        - package: package name
        - project: OBS/IBS project
        - api_url: OBS/IBS API URL
        - is_git: whether it's a git workspace
        - version: current version from spec/service file (if detectable)
        - git_branch: current git branch (if git workspace)
    """
    if not workspace_path.exists() or not workspace_path.is_dir():
        print_color(f"Error: Workspace not found: {workspace_path}", "red")
        return None

    info = {
        "package": workspace_path.name,
        "project": None,
        "api_url": None,
        "is_git": False,
        "version": None,
        "git_branch": None,
        "src_git_url": None,
    }

    # Check if it's a git repository
    git_dir = workspace_path / ".git"
    if git_dir.exists():
        info["is_git"] = True

        # Get current branch
        try:
            result = run_cmd(
                ["git", "branch", "--show-current"],
                cwd=workspace_path, timeout=10, check=False
            )
            if result.returncode == 0:
                info["git_branch"] = result.stdout.strip()
        except Exception:
            pass

        # Try to get git remote URL to identify the server
        # Check multiple remotes: upstream (preferred), origin, or any src.suse.de/src.opensuse.org remote
        for remote_name in ["upstream", "origin"]:
            try:
                result = run_cmd(
                    ["git", "remote", "get-url", remote_name],
                    cwd=workspace_path, timeout=10, check=False
                )
                if result.returncode == 0:
                    remote_url = result.stdout.strip()
                    if "src.suse.de" in remote_url or "src.opensuse.org" in remote_url:
                        info["src_git_url"] = remote_url
                        break
            except Exception:
                pass

        # For git workflow, determine API URL from the git host
        if info["src_git_url"]:
            if "src.suse.de" in info["src_git_url"]:
                info["api_url"] = IBS_API
            elif "src.opensuse.org" in info["src_git_url"]:
                info["api_url"] = OBS_API

        # Try to infer project from branch name for src-git workflows
        # Branch naming conventions:
        #   slfo-main -> SUSE:SLFO:Main
        #   slfo-1.1 -> SUSE:SLFO:Products:SLES:16.0 (approximate)
        #   factory -> openSUSE:Factory
        #   Branch might have suffix like -update-2.3.1, strip it
        if info["git_branch"] and not info["project"]:
            branch = info["git_branch"]
            # Strip common suffixes like -update-X.Y.Z or -pkg-X.Y.Z
            base_branch = re.sub(r'-(update|pkg)-[\d.]+.*$', '', branch)
            base_branch = re.sub(r'-[a-f0-9]{7,}$', '', base_branch)  # Strip commit hashes

            branch_to_project = {
                "slfo-main": "SUSE:SLFO:Main",
                "slfo-1.1": "SUSE:SLFO:Products:SLES:16.0",
                "slfo-1.2": "SUSE:SLFO:Products:SLES:16.0",
                "factory": "openSUSE:Factory",
                "tumbleweed": "openSUSE:Tumbleweed",
            }

            if base_branch in branch_to_project:
                info["project"] = branch_to_project[base_branch]

    # Check for OBS/IBS checkout (.osc directory)
    osc_dir = workspace_path / ".osc"
    if osc_dir.exists():
        # Read _apiurl file
        apiurl_file = osc_dir / "_apiurl"
        if apiurl_file.exists():
            info["api_url"] = apiurl_file.read_text().strip()

        # Read _project file
        project_file = osc_dir / "_project"
        if project_file.exists():
            info["project"] = project_file.read_text().strip()

        # Read _package file (should match directory name)
        package_file = osc_dir / "_package"
        if package_file.exists():
            info["package"] = package_file.read_text().strip()

    # Try to get version from spec file
    spec_files = list(workspace_path.glob("*.spec"))
    if spec_files:
        try:
            spec_content = spec_files[0].read_text()
            for line in spec_content.split('\n'):
                if line.strip().lower().startswith('version:'):
                    info["version"] = line.split(':', 1)[1].strip()
                    break
        except Exception:
            pass

    # Try to get version from _service file
    service_file = workspace_path / "_service"
    if service_file.exists() and not info["version"]:
        try:
            parser = ServiceFileParser(service_file)
            info["version"] = parser.get_version()
        except Exception:
            pass

    return info


def resume_from_workspace(
    workspace_path: Path,
    ai_provider: str = "claude",
) -> bool:
    """Resume work from an existing workspace.

    Prompts user for what actions to take, defaulting to N (opt-in).
    Returns True if work was completed successfully.
    """
    print_color(f"\nResuming from workspace: {workspace_path}", "blue")

    # Detect workspace info
    ws_info = detect_workspace_info(workspace_path)
    if not ws_info:
        return False

    print_color("\nDetected workspace information:", "bold")
    print(f"  Package: {ws_info['package']}")
    if ws_info["project"]:
        print(f"  Project: {ws_info['project']}")
    if ws_info["api_url"]:
        server = "IBS" if "suse.de" in ws_info["api_url"] else "OBS"
        print(f"  Server: {server} ({ws_info['api_url']})")
    print(f"  Workflow: {'Git (src-git)' if ws_info['is_git'] else 'Traditional OBS'}")
    if ws_info["git_branch"]:
        print(f"  Git branch: {ws_info['git_branch']}")
    if ws_info["version"]:
        print(f"  Current version: {ws_info['version']}")

    # Prompt for missing information needed for builds
    try:
        if not ws_info["api_url"]:
            print_color("\n  Server not detected. Please select:", "yellow")
            print("    1. IBS (api.suse.de) - internal SUSE builds")
            print("    2. OBS (api.opensuse.org) - openSUSE builds")
            choice = input("  Choice [1/2]: ").strip()
            if choice == "1":
                ws_info["api_url"] = IBS_API
            else:
                ws_info["api_url"] = OBS_API

        if not ws_info["project"]:
            print_color("\n  Project not detected.", "yellow")
            project = input("  Enter OBS/IBS project (e.g., SUSE:SLFO:Main): ").strip()
            if project:
                ws_info["project"] = project
            else:
                print_color("  Warning: No project specified. Build and submit will fail.", "yellow")
                ws_info["project"] = "unknown"
    except (KeyboardInterrupt, EOFError):
        print()
        return False

    # Create a minimal PackageInstance for the updater
    instance = PackageInstance(
        server="ibs" if ws_info["api_url"] and "suse.de" in ws_info["api_url"] else "obs",
        api_url=ws_info["api_url"] or OBS_API,
        project=ws_info["project"] or "unknown",
        package=ws_info["package"],
        version=ws_info["version"],
        src_git_url=ws_info["src_git_url"],
        src_git_branch=ws_info["git_branch"],
    )

    # Ask user for target version
    print_color("\nResume workflow (default is N for all - opt in to what you need):", "yellow")

    target_version = ws_info["version"] or ""
    try:
        new_version = input(f"\n  Target version [{target_version}]: ").strip()
        if new_version:
            target_version = new_version
    except (KeyboardInterrupt, EOFError):
        print()
        return False

    # Create the updater
    updater = PackageUpdater(instance, target_version, ai_provider)
    updater.work_dir = workspace_path
    updater.is_git_workflow = ws_info["is_git"]

    # Helper for yes/no prompts with default N
    def ask_yn(prompt: str) -> bool:
        try:
            response = input(f"\n  {prompt} [y/N]: ").strip().lower()
            return response in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print()
            raise

    # Interactive workflow - ask and execute each step immediately
    try:
        # Step 1: Update source files
        if ask_yn("Update source files (run osc service)?"):
            print_color("\n--- Updating source files ---", "bold")
            if not updater._run_osc_service():
                if not updater._confirm("Service run failed. Continue anyway?"):
                    return False

        # Step 2: Update changelog
        if ask_yn("Update changelog?"):
            print_color("\n--- Updating changelog ---", "bold")
            if not updater._create_changelog():
                if not updater._confirm("Changelog update failed. Continue anyway?"):
                    return False

        # Step 3: Test build
        build_success = True
        if ask_yn("Run test build?"):
            print_color("\n--- Running test build ---", "bold")
            build_success, build_log = updater._run_test_build()
            if not build_success:
                if updater._handle_build_failure(build_log):
                    # Retry build
                    build_success, _ = updater._run_test_build()
                if not build_success:
                    if not updater._confirm("Build failed. Continue anyway?"):
                        return False

        # Step 4: Commit (only ask if build succeeded or was skipped)
        commit_success = True
        if ws_info["is_git"]:
            if ask_yn("Commit and push changes?"):
                print_color("\n--- Committing changes ---", "bold")
                if not updater._git_commit_and_push():
                    commit_success = False
                    if not updater._confirm("Commit/push failed. Continue anyway?"):
                        return False
        else:
            if ask_yn("Commit changes to OBS?"):
                print_color("\n--- Committing changes ---", "bold")
                if not updater._osc_commit():
                    commit_success = False
                    if not updater._confirm("Commit failed. Continue anyway?"):
                        return False

        # Step 5: PR/Submit request (only ask if commit succeeded)
        if commit_success:
            if ws_info["is_git"]:
                if ask_yn("Create pull request?"):
                    print_color("\n--- Creating pull request ---", "bold")
                    if not updater._create_pull_request():
                        print_color("Pull request creation failed.", "red")
            else:
                if ask_yn("Create submit request?"):
                    print_color("\n--- Creating submit request ---", "bold")
                    if not updater._create_submit_request():
                        print_color("Submit request creation failed.", "red")

        # Step 6: Shell (always offer at the end)
        if ask_yn("Open shell in workspace?"):
            print_color("\n--- Opening shell ---", "bold")
            updater._open_workspace_shell()

        print_color("\nResume workflow completed!", "green")
        return True

    except KeyboardInterrupt:
        print_color("\n\nAborted by user.", "yellow")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="General purpose package updater for OBS/IBS (pakdev)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s himmelblau                    # Search for himmelblau package
  %(prog)s samba --ai-provider gemini    # Use Gemini for version detection
  %(prog)s himmelblau --info-only        # Just show info, don't update
  %(prog)s --resume /tmp/pkg-xxx/pkg     # Resume from existing workspace
  %(prog)s --help                        # Show this help
        """,
    )
    parser.add_argument(
        "package",
        nargs="?",  # Optional when using --resume
        help="Name of the package to search for and update",
    )
    parser.add_argument(
        "--ai-provider",
        type=str,
        default="claude",
        choices=["claude", "gemini"],
        help="AI CLI to use for version detection (default: claude)",
    )
    parser.add_argument(
        "--ai-cli-path",
        type=str,
        default=None,
        help="Path to AI CLI binary (default: same as provider name)",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip AI-based version detection",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only display package info, don't offer to update",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="Resume from an existing workspace directory (e.g., /tmp/pkg-update-xxx/package)",
    )

    args = parser.parse_args()

    print_color("=" * 70, "cyan")
    print_color("  pakdev - Package Updater", "bold")
    print_color("=" * 70, "cyan")

    # Handle resume mode - detect everything from workspace metadata
    if args.resume:
        workspace_path = Path(args.resume).resolve()
        success = resume_from_workspace(workspace_path, args.ai_provider)
        sys.exit(0 if success else 1)

    # Validate that package is provided for normal mode
    if not args.package:
        print_color("Error: Package name is required (or use --resume with a workspace path)", "red")
        sys.exit(1)

    # Initialize components
    searcher = PackageSearcher()

    # Check OBS/IBS availability
    if not searcher.obs_available:
        print_color("Warning: OBS (api.opensuse.org) not available - run 'osc -A https://api.opensuse.org whois' to authenticate", "yellow")
    if not searcher.ibs_available:
        print_color("Warning: IBS (api.suse.de) not available - check VPN, then run 'osc -A https://api.suse.de whois' to authenticate", "yellow")
    if not searcher.obs_available and not searcher.ibs_available:
        print_color("Error: Neither OBS nor IBS is available. Cannot proceed.", "red")
        sys.exit(1)

    packtrack = PacktrackClient()
    version_detector = UpstreamVersionDetector(
        ai_provider=args.ai_provider,
        ai_cli_path=args.ai_cli_path,
    )

    # Check AI availability
    if not args.skip_ai and not shutil.which(args.ai_cli_path or args.ai_provider):
        fallback = "gemini" if args.ai_provider == "claude" else "claude"
        if shutil.which(fallback):
            print_color(
                f"Note: {args.ai_provider} not found, using {fallback}", "yellow"
            )
            version_detector = UpstreamVersionDetector(
                ai_provider=fallback, ai_cli_path=fallback
            )
        else:
            print_color(
                "Note: No AI CLI available, version detection may be limited", "yellow"
            )

    # Gather information
    print_color(f"\nGathering information for: {args.package}", "bold")
    info = gather_package_info(
        args.package,
        searcher,
        packtrack,
        version_detector if not args.skip_ai else UpstreamVersionDetector(),
    )

    # Display information
    display_package_info(info)

    if args.info_only:
        return

    # Offer to update
    result = select_update_target(info, searcher)
    if not result:
        print_color("No update selected.", "yellow")
        return

    instance, version = result

    print_color(f"\nUpdating {instance.project}/{instance.package}", "green")
    if version:
        print_color(f"Target version: {version}", "green")
    else:
        print_color("Target version: (next available)", "green")

    # Check if another instance already has this version
    source_instance = offer_copy_from_existing(info, instance, version, searcher)

    # Create updater and run appropriate workflow
    updater = PackageUpdater(instance, version, args.ai_provider)

    if source_instance:
        # Copy from existing instance first
        if updater.copy_from_existing_instance(source_instance, searcher):
            print_color("\nFiles copied successfully!", "green")
            print_color("Continuing with remaining workflow steps...", "blue")
            # After copying, we still need to do changelog, build, commit, etc.
            # The copy sets up work_dir, so we can continue with a simplified workflow
            if instance.is_git_managed:
                # For git: the work_dir may not be set up yet if copy_to_git_destination
                # returned early. In that case, run the full workflow.
                if updater.work_dir:
                    updater._continue_after_copy_git()
                else:
                    updater.update_srcgit_workflow()
            else:
                # For OBS: work_dir is set up, continue with build/commit steps
                updater._continue_after_copy_obs()
        else:
            print_color("\nCopy failed, falling back to normal workflow", "yellow")
            if instance.is_git_managed:
                updater.update_srcgit_workflow()
            else:
                updater.update_obs_workflow()
    else:
        # Normal workflow
        if instance.is_git_managed:
            updater.update_srcgit_workflow()
        else:
            updater.update_obs_workflow()


if __name__ == "__main__":
    main()
