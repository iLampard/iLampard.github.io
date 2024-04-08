---
layout: page
title: archive
permalink: /archive/
description: A growing collection of notes on machine learning and mathematics.
nav: true
display_categories: [machine learning, mathematics]
nav_order: 4
#horizontal: false
---


{% if site.enable_project_categories and page.display_categories %} {% for category in page.display_categories %}
{{category}}
{% assign categorized_projects = site.projects | where: "category", category %} {% assign sorted_projects = categorized_projects | sort: "importance" %} {% if page.horizontal %}
{% for project in sorted_projects %} {% include projects_horizontal.html %} {% endfor %}
{% else %}
{% for project in sorted_projects %} {% include projects.html %} {% endfor %}
{% endif %} {% endfor %}
{% else %}

{% assign sorted_projects = site.projects | sort: "importance" %}
<!-- Generate cards for each project -->
{% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.html %}
    {% endfor %}
    </div>
  </div>
{% else %}
  <div class="grid">
    {% for project in sorted_projects %}
      {% include projects.html %}
    {% endfor %}
  </div>
{% endif %}
{% endif %}
