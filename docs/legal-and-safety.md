# Scope, safety, and legal notes

This page explains the project's intended boundary. It is not legal advice.
Laws and platform rules change, so check the current rules that apply to your
location and use case.

## Intended scope

The project removes AI provenance marks that a platform adds to content the
user generated or edited themselves. Examples include:

- visible AI generation labels;
- invisible provenance watermarks;
- C2PA and metadata based AI disclosures.

The purpose is user control over the user's own output, false positive cleanup,
interoperability work, and watermark robustness research.

## Out of scope

The project does not provide automatic removal for marks that protect a third
party's paid or copyrighted asset, including:

- stock agency previews;
- marketplace and classifieds marks;
- tiled overlays used to gate a purchase;
- artist protection systems such as Nightshade or Glaze.

The `erase` command is a generic region tool. Users are responsible for having
the right to edit the selected content.

## What removal does not prove

Removing a local signal does not:

- prove that an image is human made;
- remove server side generation history;
- anonymize the generating account;
- defeat every statistical AI detector;
- guarantee that a provider's current verifier will reject the result;
- make deceptive or unlawful use permissible.

An original file may remain linked to an account or generation session in a
provider's systems even after a local copy is changed.

## Legal context

Some jurisdictions and platforms require AI generated content to carry visible
or machine readable disclosures. Rules may apply to providers, publishers,
users, or a combination of them. Some laws also restrict removing or
suppressing provenance information.

Before removing a mark, consider:

1. whether you own or are authorized to edit the content;
2. whether a disclosure is legally required where the content will be used;
3. whether removal would mislead a viewer about authorship or origin;
4. whether copyright management information is involved;
5. whether a platform's terms prohibit the change.

The repository does not provide jurisdiction specific legal advice. The user is
responsible for checking current law and policy.

## Appropriate uses

Examples that fit the project scope include:

- removing metadata that exposes an account identifier from your own file;
- correcting an AI label applied to a human photograph after a limited edit;
- publishing your own generated artwork under the disclosure rules that apply
  to you;
- testing the robustness of watermarking and provenance systems;
- evaluating image processing pipelines in a controlled environment.

## Uses the project does not condone

- fraud or impersonation;
- nonconsensual sexual imagery;
- hiding copyright infringement;
- presenting generated content as human made where that claim is deceptive;
- evading a disclosure that the law requires;
- removing protection from someone else's paid asset.

## Reporting security or safety concerns

Open a GitHub issue when the concern can be discussed publicly without exposing
private data. Do not attach confidential images, credentials, or personal
information to a public issue.
