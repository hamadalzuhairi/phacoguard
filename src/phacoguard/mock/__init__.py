"""Annotation-driven mock demo (HANDOFF.md section 2a).

Every indicator in this package fires at a moment taken from an expert dataset
label. Nothing here infers a moment from pixels, and nothing here applies a
threshold chosen by the team. A marker with no labelled source is reported as
having no labelled source; it is never silently treated as absent.

This package makes no network calls (CLAUDE.md rule 2).
"""
