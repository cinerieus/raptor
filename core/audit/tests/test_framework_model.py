"""Tests for core.audit.framework_model."""

from __future__ import annotations

from core.audit.framework_model import (
    FrameworkGuarantee,
    format_framework_context,
    framework_negates_cwe,
)


class TestFrameworkNegatesCwe:
    def test_django_orm_negates_sqli(self):
        source = "from django.db import models\nqs = MyModel.objects.filter(name=user_input)"
        result = framework_negates_cwe("views.py", source, "CWE-89")
        assert result is not None
        assert result.framework == "django"
        assert "CWE-89" in result.negates_cwe

    def test_django_template_negates_xss(self):
        source = "from django.template import loader\nrender_to_string('t.html', ctx)"
        result = framework_negates_cwe("views.py", source, "CWE-79")
        assert result is not None
        assert "CWE-79" in result.negates_cwe

    def test_django_template_with_mark_safe_does_not_negate(self):
        source = "from django.template import loader\nmark_safe(user_input)"
        result = framework_negates_cwe("views.py", source, "CWE-79")
        assert result is None

    def test_spring_jdbc_negates_sqli(self):
        source = "JdbcTemplate tmpl = new JdbcTemplate(ds);\ntmpl.query(sql, ?);"
        result = framework_negates_cwe("Dao.java", source, "CWE-89")
        assert result is not None
        assert result.framework == "spring"

    def test_go_html_template_negates_xss(self):
        source = 'import "html/template"\nt := template.Must(template.New("").Parse(tmpl))'
        result = framework_negates_cwe("handler.go", source, "CWE-79")
        assert result is not None
        assert result.framework == "go"

    def test_django_mark_safe_before_render_does_not_negate(self) -> None:
        # The escape hatch voids the guarantee regardless of where it
        # appears relative to the rendering call.
        source = (
            "from django.shortcuts import render\n"
            "safe_body = mark_safe(user_html)\n"
            "return render(request, 't.html', {'body': safe_body})\n"
        )
        result = framework_negates_cwe("views.py", source, "CWE-79")
        assert result is None

    def test_rails_html_safe_alone_does_not_negate(self) -> None:
        # .html_safe disables escaping; it must never count as evidence
        # that escaping is in force.
        source = "raw_output = params[:name].html_safe\n"
        result = framework_negates_cwe("show.html.erb", source, "CWE-79")
        assert result is None

    def test_rails_erb_with_html_safe_does_not_negate(self) -> None:
        source = "<%= comment.body.html_safe %>\n"
        result = framework_negates_cwe("show.html.erb", source, "CWE-79")
        assert result is None

    def test_rails_erb_without_escape_hatch_negates_xss(self) -> None:
        source = "<%= user.name %>\n<%= @post.title %>\n"
        result = framework_negates_cwe("show.html.erb", source, "CWE-79")
        assert result is not None
        assert result.framework == "rails"
        assert "CWE-79" in result.negates_cwe

    def test_bare_filter_without_sqlalchemy_import_does_not_negate(self) -> None:
        # .filter( is a generic method name; without framework import
        # evidence it says nothing about SQL parameterisation.
        source = "results = queryset.filter(name=user_input)\n"
        result = framework_negates_cwe("query.py", source, "CWE-89")
        assert result is None

    def test_sqlalchemy_filter_with_import_negates_sqli(self) -> None:
        source = (
            "from sqlalchemy import select\n"
            "q = session.query(User).filter(User.name == name)\n"
        )
        result = framework_negates_cwe("query.py", source, "CWE-89")
        assert result is not None
        assert result.framework == "flask"
        assert "CWE-89" in result.negates_cwe

    def test_flask_render_template_negates_xss(self) -> None:
        source = (
            "from flask import render_template\n"
            "return render_template('index.html', name=name)\n"
        )
        result = framework_negates_cwe("app.py", source, "CWE-79")
        assert result is not None
        assert result.framework == "flask"

    def test_flask_safe_filter_does_not_negate(self) -> None:
        source = (
            "from flask import render_template_string\n"
            "return render_template_string('{{ body|safe }}', body=body)\n"
        )
        result = framework_negates_cwe("app.py", source, "CWE-79")
        assert result is None

    def test_no_framework_returns_none(self):
        source = "int main() { return 0; }"
        result = framework_negates_cwe("main.c", source, "CWE-89")
        assert result is None

    def test_unrelated_cwe_returns_none(self):
        source = "from django.db import models"
        result = framework_negates_cwe("views.py", source, "CWE-22")
        assert result is None


class TestFormatFrameworkContext:
    def test_empty_list(self):
        assert format_framework_context([]) == ""

    def test_single_guarantee(self):
        g = FrameworkGuarantee(
            framework="django",
            pattern="ORM .filter()",
            guarantees="parameterises queries",
            negates_cwe=["CWE-89"],
        )
        text = format_framework_context([g])
        assert "Django" in text
        assert "CWE-89" in text
        assert "Framework conventions:" in text
