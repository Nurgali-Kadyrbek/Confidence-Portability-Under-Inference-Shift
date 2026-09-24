# MDPI LaTeX template provenance

| field | value |
| --- | --- |
| Source page | https://www.mdpi.com/authors/latex (official MDPI author instructions) |
| File URL | https://res.mdpi.com/data/MDPI_template_ACS.zip?v=20260911 |
| Resolved host | https://mdpi-res.com/data/MDPI_template_ACS.zip?v=20260911 |
| Template version tag | v=20260911 |
| MDPI "last updated" | 11 September 2026 |
| Retrieved (UTC) | 2026-09-23T01:20:31Z |
| Archive SHA-256 | `62744425fbcec9cd3e58147cbee65bdec2e2ff0440f29792c26edc97a11b6c70` |
| Archive size | 671712 bytes |

MDPI publishes three variants of the same template that differ only in the
bibliography style: ACS (the MDPI default numbered style), APA and Chicago.
`Information` uses MDPI's default numbered citation style, so the ACS archive
is the correct one; it also ships `mdpi_apacite.bst` and `mdpi_chicago.bst`
alongside `mdpi.bst`.

The class infrastructure in `Definitions/` is used unmodified. No file in
`Definitions/` is edited.

MDPI's web front end rejects non-browser clients with HTTP 403 (Akamai). The
archive was retrieved with standard browser request headers from MDPI's own
resource host; no third-party mirror or GitHub copy was used.
