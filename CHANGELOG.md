# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and follows the Sionna versioning:
`(sionna_major, sionna_minor, sionna_patch, fix)`.

The three first parts indicate the official supported version of
Sionna, and `fix` is used to indicate patch version
for our library.

<!-- start changelog -->

(unreleased)=
## [Unreleased](https://github.com/jeertmans/sionna-vispy/compare/v1.1.0...HEAD)

(unreleased-added)=
### Added

- Added `get_canvas(scene: Scene) -> SceneCanvas` function
  to retrieve the VisPy canvas from a Sionna scene.
- Added warning for unimplemented point picking and clipping plane slider features.
  [#26](https://github.com/jeertmans/sionna-vispy/pull/26)
  [#26](https://github.com/jeertmans/sionna-vispy/pull/26)

(unreleased-chore)=
### Chore

- Added missing `py.typed` file.
  [#25](https://github.com/jeertmans/sionna-vispy/pull/25)
- Bumped compatibility with Sionna v1.2.0.
  [#26](https://github.com/jeertmans/sionna-vispy/pull/26)

(v1.1.0)=
## [v1.1.0](https://github.com/jeertmans/sionna-vispy/compare/v1.0.0...v1.1.0)

(v1.1.0-chore)=
### Chore

- Bumped compatibility with Sionna v1.1.0.

(v1.0.0)=
## [v1.0.0](https://github.com/jeertmans/sionna-vispy/compare/v0.19.0...v1.0.0)

(v1.0.0-fixed)=
### Fixed

- Fixed clipping plane (now works as expected).
  [#13](https://github.com/jeertmans/sionna-vispy/pull/13)

(v1.0.0-refactored)=
### Refactored

- Refactored library to be compatible with Sionna v1,
  except for **show_legend** method that is not yet implemented
  (any contribution is welcome).
  [#13](https://github.com/jeertmans/sionna-vispy/pull/13)

(v0.19.0)=
## [v0.19.0](https://github.com/jeertmans/sionna-vispy/compare/v0.18.0.1...v0.19.0)

(v0.19.0-chore)=
### Chore

- Changed project manager from Rye to uv.
  [#5](https://github.com/jeertmans/sionna-vispy/pull/5)

(v0.19.0-fixed)=
### Fixed

- Disconnect ray paths so they are drawn separately.
  [#4](https://github.com/jeertmans/sionna-vispy/pull/4)

(v0.18.0.1)=
## [v0.18.0.1](https://github.com/jeertmans/sionna-vispy/compare/v0.18.0...v0.18.0.1)

(v0.18.0-chore)=
### Chore

- Added project URLs to PyPI.

(v0.18.0)=
## [v0.18.0](https://github.com/jeertmans/sionna-vispy/commits/v0.18.0)

(v0.18.0-added)=
### Added

- Created first package.
  [#1](https://github.com/jeertmans/sionna-vispy/pull/1)

<!-- end changelog -->
