---
name: Eruption Forecast
description: A light scientific workspace for reproducible seismic forecasting.
colors:
  forest: "#32634e"
  forest-hover: "#244c39"
  paper: "#f3f4ef"
  surface: "#fff"
  ink: "#27362d"
  muted: "#58685e"
  line: "#d7ded5"
  sidebar: "#e9ede5"
  toolbar: "#fafbf7"
  navigation-hover: "#dce4d7"
  navigation-selected: "#d3dfcd"
  navigation-selected-text: "#244932"
  input-line: "#cfd8ca"
  error-surface: "#fff0e9"
  error-text: "#943d26"
  error-line: "#ebc1b3"
  success-surface: "#e2edda"
  success-text: "#3f6632"
  running-surface: "#e1edf0"
  running-text: "#2c6674"
  warning-surface: "#f3ead2"
  warning-text: "#81612c"
typography:
  headline:
    fontFamily: '"Source Serif 4", serif'
    fontSize: "clamp(28px, 3vw, 36px)"
    fontWeight: 400
    lineHeight: 1.25
    letterSpacing: "-.025em"
  title:
    fontFamily: '"Inter", sans-serif'
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1.45
  body:
    fontFamily: '"Inter", sans-serif'
    fontSize: "13px"
    lineHeight: 1.75
  label:
    fontFamily: '"Inter", sans-serif'
    fontSize: "12px"
    fontWeight: 500
  code:
    fontFamily: "ui-monospace, Consolas, monospace"
    fontSize: "11px"
rounded:
  input: "4px"
  control: "5px"
  panel: "6px"
spacing:
  panel: "25px"
  mobile-panel: "20px"
  button: "10px 15px"
  input: "9px 11px"
components:
  button-primary:
    backgroundColor: "{colors.forest}"
    textColor: "{colors.surface}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "{spacing.button}"
  button-primary-hover:
    backgroundColor: "{colors.forest-hover}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.input}"
    padding: "{spacing.input}"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel}"
---

# Design System: Eruption Forecast

## Overview

The implemented interface is a light scientific workspace: restrained forest accents, warm pale backgrounds, dense expert forms, and persistent navigation. This document captures the incumbent implementation in `app/frontend/src/styles/globals.css`; the frontmatter is the normative token layer. No separate creative metaphor has been established.

Serif page headings give the workspace a reading hierarchy while compact sans-serif controls support repeated configuration. Scientific artifacts remain the focus of result viewers. Status labels combine text with color.

Key characteristics:

- Light surfaces with muted green structure.
- Compact, explicit settings and persistent workflow context.
- Flat panels, fine borders, and visible keyboard focus.

## Colors

### Primary

Forest identifies primary actions, links, selected tabs, checkbox accents, and keyboard focus. Forest hover deepens primary buttons during interaction.

### Neutral

Paper is the application background; surface is the white field and panel background. Ink carries primary content; muted carries explanatory text. Line defines structural separation. Sidebar and toolbar distinguish persistent navigation and configuration context without elevated cards. Navigation hover and selected colors show location within the workflow.

### Semantic

Error colors frame actionable failures. Success, running, and warning colors distinguish job states; visible text always identifies the state. These are functional status colors, rather than additional brand accents.

## Typography

Page headings use Source Serif 4 with a serif fallback. Body, navigation, labels, and buttons use Inter with a sans-serif fallback. Configuration, logs, and raw artifact values use the system monospace stack.

The headline is responsive; title and label roles stay compact. Body paragraphs have a maximum measure of 75 characters. Page descriptions use smaller text and a 72-character maximum. Numeric context and tables use tabular figures to keep values aligned. Do not apply the large page-heading role to individual form sections.

## Layout

The desktop shell pairs a sticky navigation column (218px) with flexible content. A topbar (66px high) and configuration bar remain above the page. Main content has a maximum width of 1600px and desktop padding of 38px 40px 60px.

Expert forms use two equal columns, with a vertical gap of 23px and horizontal gap of 30px. Wide fields and parameter groups span both columns. Scenario, job, and result pages use a list beside a flexible detail panel. Overflowing data tables and artifact lists scroll inside their own containers.

At 1100px, main horizontal padding reduces and ancillary workflow context disappears. At 800px, navigation becomes an off-canvas drawer, detail layouts stack, and panels use mobile padding. At 480px, form fields become one column and toolbars wrap. At 1600px and above, form gaps increase. Preserve the actual breakpoint behavior when extending the shell.

## Elevation & Depth

Depth comes mainly from pale surface changes and fine borders. Panels and forms have no box shadow. A toast uses a small shadow to separate transient feedback from content; the mobile navigation drawer uses a lateral shadow. Avoid adding shadows to ordinary scientific content panels.

Motion supports state changes: buttons and links transition color and background over 150ms; the drawer moves over 200ms; toast arrival lasts 200ms with ease-out. The reduced-motion media query disables animation and transitions.

## Shapes

Fields and status labels have compact corners; buttons and navigation items use slightly softer corners; content panels share the largest standard corner. Borders are generally one pixel. Icons, checkboxes, and progress bars remain small. The connection indicator is circular and carries a textual connection description beside it.

## Components

### Buttons

Primary buttons use forest with white text; secondary buttons use white, muted green text, and a fine green border. Both use the standard button padding, a minimum height of 38px, and medium-weight label text. Icon buttons are 34px square. Disabled buttons reduce opacity to 0.55 and use a not-allowed cursor. Keyboard focus has a two-pixel forest outline with a three-pixel offset.

### Inputs / Fields

Inputs use white surfaces, ink text, input-line borders, and a minimum height of 37px. Labels precede controls; smaller hints appear beneath them. Textareas use monospace. Checkboxes use forest accents. Reset controls sit beside their fields. Field errors use the semantic error text and keep the explanation explicit.

### Cards / Containers

Scenario editors, job details, and artifact viewers use white surfaces, line borders, panel corners, and standard panel padding. Form panels use 27px padding on desktop. These containers support content grouping; their default treatment stays flat.

### Navigation

Sidebar links pair icons with text and use compact vertical rhythm. Hover uses the navigation-hover surface. Active links use navigation-selected, selected text, and semibold weight. Tabs use an underline and forest text for the active state. On small screens, a menu button exposes the existing sidebar drawer.

### Status and Artifact Viewing

Job statuses use compact, text-bearing labels. Scientific figures scale to the viewer width without stretching their aspect ratio. CSV tables use sticky headers, tabular numerals, and horizontal scrolling; logs retain line breaks and wrap long paths. Empty states offer a concise explanation and relevant action.

## Do's and Don'ts

- Do reuse forest for actions, selection, and focus.
- Do preserve visible labels, hints, and textual job states.
- Do use the serif headline for page identity and compact sans-serif roles for controls.
- Do keep scientific plots, tables, timestamps, and configuration values faithful to their source.
- Don't add shadows to ordinary panels.
- Don't rely on color alone to explain job state.
- Don't remove the reduced-motion treatment or visible keyboard focus when adding interactions.
