# Retinal Captioning Enhancement Plan
## Dissertation Extension — Option 1: Captioning Module
### Update for Current Retinal XAI + Mistral System
### No Code — Architecture and Integration Plan Only

---

# Overview

## Purpose
This enhancement adds a retinal captioning module to the current system.
Instead of sending only numerical outputs to Mistral, the system will also
produce a short textual description of retinal findings extracted directly
from the fundus image.

## Why This Extension Matters
The current architecture uses:
- ResNet-50 for diabetic retinopathy grading
- EfficientNet-B4 for cardiovascular risk scoring
- Mistral-7B for narrative report generation

However, both image models compress complex retinal patterns into single scores.
This means clinically useful visual detail is lost before the report is generated.
A captioning module preserves those details by converting retinal patterns into
plain-language descriptions that can be included in the LLM prompt.

## Core Idea
The retinal image will now produce three forms of evidence:
1. DR grade
2. CVD risk score
3. Retinal feature caption

The caption is then sent to BioMistral or Mistral together with the structured
scores and optional patient clinical history.
This creates a lightweight vision-language pipeline without replacing the
existing system.

---

# Updated System Architecture

## Current Pipeline
Retinal image → preprocessing → ResNet-50 → DR grade
Retinal image → preprocessing → EfficientNet-B4 → CVD score
Scores → prompt builder → Mistral-7B → report

## Proposed Pipeline
Retinal image → preprocessing → ResNet-50 → DR grade
Retinal image → preprocessing → EfficientNet-B4 → CVD score
Retinal image → captioning encoder-decoder → retinal caption
Retinal caption + scores + optional clinical history → prompt builder → BioMistral or Mistral → richer report

## What the Caption Adds
The caption can express findings such as:
- Mild arteriovenous nicking
- Increased vessel tortuosity
- Microaneurysms present
- Haemorrhages absent
- Optic disc appears normal
- Exudates not visible

These details are clinically meaningful and are easier for the LLM to reason over
than raw hidden features alone.

---

# Model Design

## Recommended Architecture
Use a lightweight image captioning design:
- Encoder: ResNet-50 backbone
- Decoder: small LSTM or compact Transformer decoder
- Output: 1 to 3 short clinical sentences

## Why This Is the Best Fit
This approach is achievable within the current project scope.
It does not require replacing the existing DR or CVD models.
It also fits naturally into the dissertation as a modular enhancement rather than
an entirely new system.

## Suggested Caption Style
The model should generate concise factual descriptions, for example:

"Fundus image shows mild vascular tortuosity with scattered microaneurysms.
No obvious haemorrhages or hard exudates are visible. Optic disc appearance
is grossly preserved."

The target style should be descriptive rather than diagnostic.
This reduces hallucination risk and keeps the caption grounded in visual evidence.

---

# Dataset Strategy

## Practical Dataset Plan
Use a retinal dataset with disease labels and convert labels into templated captions.
This is the most realistic approach if true image-report pairs are limited.

## Label-to-Text Conversion
If an image has labels such as:
- microaneurysm
- hard exudates
- tortuous vessels
- disc abnormality

Then generate a training caption such as:
"Fundus image shows microaneurysms, hard exudates, and vascular tortuosity.
Optic disc abnormality may be present."

## Why Templated Captions Make Sense
True ophthalmology report datasets are harder to obtain and clean.
Templated captions allow supervised training using existing annotated retinal datasets.
They also make the training process manageable for dissertation timelines.

---

# Integration with Current Plan

## Step 1 — New Module
Add a new folder or module for retinal captioning.
Suggested file names:
- ml/retinal_caption_model.py
- ml/caption_inference.py
- data/caption_templates.py

## Step 2 — Training Output
Train the captioning model separately from the current DR and CVD models.
Export the best checkpoint and load it during inference in the Flask app.

## Step 3 — Inference Flow
After preprocessing the uploaded image:
1. Run DR model
2. Run CVD model
3. Run captioning model
4. Return the caption text
5. Pass all outputs into the prompt builder

## Step 4 — Prompt Builder Update
Update the LLM prompt to include a new section:

"Retinal feature description:
[CAPTION OUTPUT HERE]"

This should appear before the final instruction asking BioMistral or Mistral
to generate the report.

## Step 5 — Results Page Update
Add a new card on the results page titled:
"Retinal Feature Summary"

Display the generated caption in a highlighted text box.
This makes the new module visible to the user and useful for dissertation screenshots.

---

# Role of BioMistral

## Why Send the Caption to BioMistral
BioMistral can use structured medical text more effectively than isolated scores.
A retinal caption gives it intermediate semantic evidence that resembles clinical notes.
This improves the coherence and clinical relevance of the generated report.

## Prompt Structure Example
Include the following fields in the report prompt:
- DR grade and label
- CVD score
- Fused score
- Clinical context score if available
- Patient free-text history if available
- Retinal feature caption

## Expected Benefit
Instead of writing a report from only numbers, the LLM can reason over both
quantitative outputs and descriptive retinal findings.
This should produce more natural and medically informative explanations.

---

# Fusion Strategy

## Important Design Choice
The caption should not directly replace the fusion formula.
It should act as an explanatory and contextual input to the LLM layer.

## Why This Is Better
The current numerical fusion remains interpretable and easy to justify.
The caption adds semantic richness without making the risk score harder to audit.
This separation is useful for dissertation defensibility.

## Optional Future Extension
In a later phase, the captioning model or its encoder embeddings could be converted
into a structured retinal feature score and added to the fusion formula.
For now, keep the caption in the report-generation pathway only.

---

# Validation Plan

## What to Evaluate
Evaluate the captioning module on three levels:
1. Caption quality
2. Clinical usefulness
3. Effect on final LLM report quality

## Caption Quality Checks
Assess whether the generated caption correctly mentions visible findings such as:
- microaneurysms
- exudates
- vessel tortuosity
- haemorrhages
- disc appearance

## Report Quality Checks
Compare two report versions for the same image:
- Without caption input
- With caption input

Assess whether the caption-enhanced version is more specific, more clinically
informative, and better aligned with visible retinal features.

---

# Dissertation Write-Up

## Methodology Addition

"To enrich the semantic information available to the language model, a retinal
captioning module was proposed as an extension to the baseline score-fusion
architecture. The module uses an encoder-decoder design, with a ResNet-50 image
encoder and a lightweight language decoder, to generate short descriptive text
summarising visible retinal findings from the uploaded fundus image. This text is
then injected into the BioMistral prompt alongside the diabetic retinopathy grade,
cardiovascular risk score, fused score, and optional free-text clinical history.
The objective is not to replace the existing predictive models, but to preserve
intermediate visual evidence in a form that can be more effectively interpreted by
a biomedical language model."

## Rationale Addition

"The motivation for this extension is that classification models compress complex
retinal morphology into single numerical outputs, which may omit clinically useful
detail. A captioning layer allows features such as vascular tortuosity,
arteriovenous nicking, haemorrhages, exudates, and optic disc appearance to be
expressed in natural language. This more closely reflects the descriptive process
used by human clinicians and supports the transition from a simple multimodel
fusion pipeline toward a lightweight vision-language architecture."

## Limitations Addition

"The proposed captioning model is intended as a descriptive support layer rather
than an autonomous diagnostic engine. Its output depends on the quality of image
annotations and may inherit bias from templated training captions. Accordingly,
the generated text should be treated as contextual evidence for report generation,
not as a standalone diagnosis."

---

# Recommended Update to Current Plan

Add the following new section to the main implementation plan:

## New Enhancement — Retinal Captioning Module
A lightweight retinal captioning model will be added to the existing retinal XAI
system to convert visible fundus features into short clinical descriptions.
The model will use a ResNet-50 encoder and a compact text decoder to generate
captions such as microaneurysms present, mild vessel tortuosity, or no obvious
haemorrhages. The generated caption will be inserted into the BioMistral prompt
alongside the DR grade, CVD score, fused score, and optional patient clinical
history. This extension preserves interpretable visual evidence that would
otherwise be compressed into numerical outputs alone, thereby improving the
semantic context available for LLM-based report generation while keeping the
existing fusion formula unchanged.

---

# Antigravity Update Block

Paste this into your existing Antigravity plan as an additional task block:

TASK: Add Retinal Captioning Module to Current Retinal XAI System

Add a lightweight retinal image captioning module to the current pipeline.
Use a ResNet-50 encoder and a small LSTM or compact Transformer decoder to
generate short retinal feature descriptions from the uploaded fundus image.
Train the captioning module separately using retinal labels converted into
templated captions if full report pairs are unavailable. During inference,
run the captioning model after image preprocessing and pass the generated
caption into the BioMistral or Mistral prompt builder alongside the DR grade,
CVD score, fused score, and any optional clinical history text. Do not change
the existing numerical fusion formula at this stage; use the caption only as a
semantic support layer for report generation. Update the results page to show
a new Retinal Feature Summary card containing the generated caption. Add this
module as a dissertation extension toward a lightweight vision-language medical
AI architecture.

---

# Implementation Priority

## Phase 1
Keep the text-box clinical history enhancement as the main completed fix.
This solves the immediate false-negative problem for non-retinal cardiac history.

## Phase 2
Add the retinal captioning module as the next research extension.
This strengthens the report-generation quality and increases the originality of
the dissertation.

## Phase 3
Optionally compare Mistral and BioMistral using the same caption-enhanced prompt.
This can be presented as an ablation or comparative language-model evaluation.
