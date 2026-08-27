"""Convert supported legal source files into auditable structured Markdown.

This package belongs to the offline half of the project.  It deliberately does
not contain vector-store or API code: conversion should be inspectable and
repeatable before any text is embedded.
"""

from offline_ingestion.markdown_converter.converter import convert_document

__all__ = ["convert_document"]
