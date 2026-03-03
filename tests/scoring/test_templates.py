"""Tests for TemplateRenderer — judge prompt template rendering."""

from __future__ import annotations

import pytest
from jinja2 import TemplateNotFound, UndefinedError

from saber.scoring.templates import TemplateRenderer


class TestTemplateRendererConstruction:
    """TemplateRenderer can be constructed in dict, file, or empty mode."""

    def test_construct_with_dict_templates(self) -> None:
        renderer = TemplateRenderer(templates={"hello.j2": "Hello {{ name }}"})
        assert isinstance(renderer, TemplateRenderer)

    def test_construct_with_file_templates(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        (d / "greet.j2").write_text("Hi {{ who }}")
        renderer = TemplateRenderer(templates_dir=d)
        assert isinstance(renderer, TemplateRenderer)

    def test_construct_with_neither(self) -> None:
        renderer = TemplateRenderer()
        assert isinstance(renderer, TemplateRenderer)


class TestTemplateRendererRender:
    """TemplateRenderer.render produces expected output."""

    def test_render_simple_template(self) -> None:
        renderer = TemplateRenderer(templates={"msg.j2": "Hello {{ name }}!"})
        result = renderer.render("msg.j2", {"name": "Alice"})
        assert result == "Hello Alice!"

    def test_render_missing_variable_raises(self) -> None:
        renderer = TemplateRenderer(templates={"t.j2": "{{ missing_var }}"})
        with pytest.raises(UndefinedError):
            renderer.render("t.j2", {})

    def test_render_missing_template_raises(self) -> None:
        renderer = TemplateRenderer(templates={})
        with pytest.raises(TemplateNotFound):
            renderer.render("no_such.j2", {})

    def test_render_complex_context(self) -> None:
        tpl = "{% for item in items %}{{ item.name }},{% endfor %}"
        renderer = TemplateRenderer(templates={"list.j2": tpl})
        result = renderer.render(
            "list.j2",
            {"items": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        )
        assert result == "a,b,c,"

    def test_render_nested_dict_context(self) -> None:
        tpl = "{{ data.level1.level2 }}"
        renderer = TemplateRenderer(templates={"nested.j2": tpl})
        result = renderer.render("nested.j2", {"data": {"level1": {"level2": "deep"}}})
        assert result == "deep"

    def test_render_from_file(self, tmp_path: object) -> None:
        from pathlib import Path

        d = Path(str(tmp_path))
        (d / "test.j2").write_text("Value: {{ val }}")
        renderer = TemplateRenderer(templates_dir=d)
        result = renderer.render("test.j2", {"val": 42})
        assert result == "Value: 42"

    def test_render_keeps_trailing_newline(self) -> None:
        renderer = TemplateRenderer(templates={"nl.j2": "line\n"})
        result = renderer.render("nl.j2", {})
        assert result.endswith("\n")

    def test_empty_loader_raises_on_any_template(self) -> None:
        renderer = TemplateRenderer()
        with pytest.raises(TemplateNotFound):
            renderer.render("anything.j2", {})
