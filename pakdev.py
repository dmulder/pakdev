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

        # Remove old tarballs if we're updating to a specific version
        if self.target_version:
            for tarball in self.work_dir.glob("*.tar.*"):
                # Check if it's an old version tarball
                if self.target_version not in tarball.name:
                    try:
                        tarball.unlink()
                        cleaned.append(tarball.name)
                    except Exception as e:
                        print_color(f"  Warning: Could not remove {tarball.name}: {e}", "yellow")

            # Also check for .tgz files
            for tarball in self.work_dir.glob("*.tgz"):
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
        return self._confirm("Re-verify the version?")

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

    def _create_changelog(self, message: Optional[str] = None) -> bool:
        """Create a changelog entry using osc vc."""
        if not self.work_dir:
            return False

        if not message:
            if self.target_version:
                message = f"Update to version {self.target_version}"
            else:
                message = "Update to latest version"

        print_color(f"\nCreating changelog entry: {message}", "blue")

        try:
            # osc vc with -m for non-interactive mode
            run_cmd(
                ["osc", "vc", "-m", message],
                cwd=self.work_dir,
                timeout=30,
            )
            print_color("  Changelog entry created", "green")
            return True
        except Exception as e:
            print_color(f"  Failed to create changelog: {e}", "yellow")
            print_color("  You may need to run 'osc vc' manually", "yellow")
            return True  # Don't fail the whole process

    def _run_test_build(self, repo: str = "openSUSE_Tumbleweed", arch: str = "x86_64") -> tuple[bool, str]:
        """Run a local test build.

        Returns (success, build_log) tuple.
        """
        if not self.work_dir:
            return False, ""

        print_color(f"\nRunning test build ({repo}/{arch})...", "blue")
        print_color("  This may take a while. Press Ctrl+C to skip.", "yellow")

        build_log_lines = []

        try:
            cmd = ["osc", "build", "--no-verify", repo, arch]

            # For git workflow, use alternative project
            if self.is_git_workflow:
                cmd = ["osc", "build", "--no-verify", f"--alternative-project={self.instance.project}", repo, arch]

            process = subprocess.Popen(
                cmd,
                cwd=self.work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output and capture it
            for line in process.stdout:
                print(line, end="")
                build_log_lines.append(line)

            process.wait(timeout=self.TIMEOUT_BUILD)

            build_log = "".join(build_log_lines)

            if process.returncode == 0:
                print_color("\n  Build succeeded!", "green")
                return True, build_log
            else:
                print_color("\n  Build failed", "red")
                return False, build_log

        except subprocess.TimeoutExpired:
            process.kill()
            print_color("\n  Build timed out", "red")
            return False, "".join(build_log_lines)
        except KeyboardInterrupt:
            process.kill()
            print_color("\n  Build skipped by user", "yellow")
            return True, ""  # User chose to skip, not a failure
        except Exception as e:
            print_color(f"\n  Build error: {e}", "red")
            return False, str(e)

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
            # Internal src.suse.de - direct push
            print_color("\nPushing to src.suse.de...", "blue")
            try:
                run_cmd(
                    ["git", "push"],
                    cwd=self.work_dir,
                    timeout=120,
                    capture=False,
                )
                print_color("  Pushed successfully", "green")
                return True
            except Exception as e:
                print_color(f"  Push failed: {e}", "red")
                return False
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
        changelog_msg = f"Update to version {self.target_version}" if self.target_version else "Update to latest version"
        custom_msg = self._prompt_input("  Changelog message", changelog_msg)
        self._create_changelog(custom_msg)

        # Step 7: Test build (optional)
        if self._confirm("Run a test build?"):
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

    def update_srcgit_workflow(self) -> bool:
        """Update package using src-git workflow."""
        print_color("\n=== src-git Workflow ===", "cyan")
        self.is_git_workflow = True

        git_url = self.instance.src_git_url
        if not git_url:
            print_color("  No src-git URL detected", "red")
            return False

        print_color(f"\nGit repository: {git_url}", "blue")
        if self.instance.src_git_branch:
            print_color(f"Branch: {self.instance.src_git_branch}", "blue")

        # Determine if internal or external
        is_internal = SRC_SUSE in git_url

        # Step 1: Clone
        print_color("\nStep 1: Cloning repository...", "blue")
        if not self._confirm("Proceed with cloning?"):
            return False

        self.work_dir = Path(tempfile.mkdtemp(prefix="pkg-update-"))
        pkg_dir = self.work_dir / self.instance.package

        try:
            run_cmd(
                ["git", "clone", git_url, str(pkg_dir)],
                timeout=120,
            )
            self.work_dir = pkg_dir

            if self.instance.src_git_branch:
                run_cmd(
                    ["git", "checkout", self.instance.src_git_branch],
                    cwd=self.work_dir,
                    timeout=30,
                )

            print_color(f"  Cloned to: {self.work_dir}", "green")
        except Exception as e:
            print_color(f"  Clone failed: {e}", "red")
            return False

        # Step 2: Update _service file
        print_color("\nStep 2: Updating _service file...", "blue")
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

        # Step 3: Run osc service (if _service file exists)
        has_service_file = (self.work_dir / "_service").exists()
        if has_service_file:
            if self._confirm("Run osc service to fetch sources?", default_yes=True):
                if not self._run_osc_service():
                    if not self._confirm("Service failed. Continue anyway?"):
                        return False

        # Step 4: Verify version matches (always do this)
        print_color("\nStep 4: Verifying version...", "blue")
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

        # Step 5: Create changelog
        changelog_msg = f"Update to version {self.target_version}" if self.target_version else "Update to latest version"
        custom_msg = self._prompt_input("  Changelog message", changelog_msg)
        self._create_changelog(custom_msg)

        # Step 6: Test build (optional)
        if self._confirm("Run a test build?"):
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

        # Step 7: Commit and push/PR
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

    # Detect git workflow for each instance
    print_color("\nDetecting workflows...", "blue")
    for instance in info.instances:
        searcher.detect_git_workflow(instance)
        if instance.is_git_managed:
            print_color(f"  {instance.project}: git-managed", "green")
        else:
            print_color(f"  {instance.project}: traditional OBS", "blue")

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
            ver = f" v{inst.version}" if inst.version else ""
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


def main():
    parser = argparse.ArgumentParser(
        description="General purpose package updater for OBS/IBS (pakdev)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s himmelblau                    # Search for himmelblau package
  %(prog)s samba --ai-provider gemini    # Use Gemini for version detection
  %(prog)s himmelblau --info-only        # Just show info, don't update
  %(prog)s --help                        # Show this help
        """,
    )
    parser.add_argument(
        "package",
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

    args = parser.parse_args()

    print_color("=" * 70, "cyan")
    print_color("  pakdev - Package Updater", "bold")
    print_color("=" * 70, "cyan")

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

    # Create updater and run appropriate workflow
    updater = PackageUpdater(instance, version, args.ai_provider)

    if instance.is_git_managed:
        updater.update_srcgit_workflow()
    else:
        updater.update_obs_workflow()


if __name__ == "__main__":
    main()
