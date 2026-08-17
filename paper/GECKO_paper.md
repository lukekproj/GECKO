---
title: 'GECKO: A researcher-in-the-loop tool for gaze event classification in Kinarm robotic experiments'
tags:
  - Python
  - neuroscience
  - eye tracking
  - gaze classification
  - sensorimotor control
  - KINARM
authors:
  - name: Luke Kroon
    orcid: 0009-0004-1849-3680
    affiliation: 1
  - name: Angus P. Muttee
    orcid: 0009-0003-3006-358X
    affiliation: 1
  - name: Blake A. Hollinger
    orcid: 0009-0000-0983-4130
    affiliation: 1
  - name: Tarkeshwar Singh
    orcid: 0000-0001-7051-6529
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - index: 1
    name: Department of Kinesiology, The Pennsylvania State University, University Park, PA-16802, USA
    ror: "04p491231"
  - index: 2
    name: Penn State Neuroscience Institute, The Pennsylvania State University, University Park, PA-16802, USA
date: 17 June 2026
bibliography: paper.bib
---

# Summary

GECKO (Gaze Event Classification in the Kinarm, Open-access) is an open-source
Python desktop application for visualizing, annotating, and exporting
eye-tracking and movement data from experiments run on Kinarm robots
[@scott1999]. Motor control studies performed with the Kinarm require
participants to interact with visual stimuli presented in a transverse plane,
which poses distinctive challenges for computing gaze kinematics and for the
subsequent identification of gaze events — moments when the eye holds still
(fixations), follows a moving object (smooth pursuits), or jumps rapidly between
points (saccades). Even with the equations proposed by @singh2016 to obtain gaze
kinematics from gaze point-of-regard (POR) data, classifying these events remains
an error-prone process. GECKO addresses this by presenting each experimental
trial as an interactive plot, cleaning and filling missing samples, allowing a
researcher to mark gaze events with a few mouse clicks at the resolution of
individual frames, and exporting the result as analysis-ready tables. By keeping
a human in the loop while automating the tedious data handling, it provides a
gold standard for gaze event classification and produces a transparent,
reproducible record of every processing decision. The tool is already in use by
multiple laboratories studying motor control in healthy humans and in patients
with neurological conditions.

# Statement of need

Kinarm robotic exoskeleton and endpoint systems are widely used in sensorimotor
neuroscience and rehabilitation to measure upper-limb movement with high spatial
and temporal precision. These systems are increasingly paired with eye trackers
(for example the EyeLink 1000) to study eye–hand coordination during reaching
and interception. The manufacturer's data files (`.kinarm`) store synchronized
gaze and kinematic channels, but provide no facility for classifying gaze
events, and two preprocessing steps must be completed before any classification
— manual or automatic — can begin.

First, gaze lost to blinks or to hardware tracking failures is recorded as
large sentinel values. These must be detected and
replaced, and the resulting missing segments interpolated, before the signal can
be differentiated. Second, the eye tracker outputs point-of-regard (POR) data as
$(x, y)$ coordinates in the transverse plane
(\autoref{fig:geometry}\,B), which must be converted into angular
coordinates. Because that plane sits in front of and below the participant, the
viewing distance varies across the workspace and the conversion is not the
simple scalar calculation available for a frontoparallel display
(\autoref{fig:geometry}\,A); recovering ocular kinematics is instead an
inherently three-dimensional geometric problem.

![**Gaze geometry in a Kinarm workspace is an inherently three-dimensional problem, and GECKO is built around it.** (A) In standard eye tracking, stimuli appear on a frontoparallel display at an approximately fixed viewing distance $b$, so a stimulus of extent $a$ subtends a single scalar visual angle, $\tan(\beta/2) = a/(2b)$. (B) In Kinarm experiments, stimuli are presented in a transverse plane in front of and below the participant. Following @singh2016, the eye is modeled as a point source at the origin of an eye-based frame $X'Y'Z'$ whose $Z'$ axis points downward, so the stimulus plane lies at $z' = H$, the eye height above the plane. Viewing distance varies across the workspace and gaze acquires a depth component, so the recorded two-dimensional point-of-regard (POR) must be transformed into eye-centered spherical coordinates: the radial eye-to-POR distance $\rho$, the azimuth $\theta$ measured in the $X'Y'$ plane from the $X'$ axis (shown at the foot of the perpendicular, which is equivalent because translation along $Z'$ leaves $x'$ and $y'$ unchanged), and the elevation $\varphi$ measured from the $+Z'$ axis. (C) GECKO implements this transformation, derives gaze angular velocity from it, and couples the result to a researcher-in-the-loop interface that assigns a saccade, pursuit, or fixation label to every frame — the step that no existing screen-based event detector performs on Kinarm data.\label{fig:geometry}](kinarm_gaze_geometry.jpg){width="100%"}

@singh2016 introduced a geometric method for computing ocular kinematics and
classifying gaze events specifically for robotic environments with moving
targets. However, no maintained, accessible software implementation of this
method exists; individual laboratories reimplement it in ad hoc scripts, which
is duplicative and hampers reproducibility. A second need concerns the labels
themselves: manual classification by an expert annotator is widely treated as
the ground truth for evaluating eye-movement event detection [@startsev2023],
yet producing such labels at scale is what existing tooling does not support. GECKO 
addresses both needs. It reads `.kinarm` files directly and implements the @singh2016 
spherical-coordinate transform, angular-velocity, and foveal-visual-radius (FVR) computations. 
The FVR accounts for the fact that, at larger viewing distances, the same angular extent spans a larger Euclidean
distance in the workspace — a correction that matters when deciding whether
gaze is close enough to a target to count as fixating or pursuing it. GECKO
then surfaces these computations in a labeling interface designed for imperfect
gaze data, making expert annotation practical across a full dataset
(\autoref{fig:geometry}\,C). The target users are motor-control,
sensorimotor-neuroscience, and rehabilitation researchers who collect gaze data
alongside Kinarm kinematics and need a consistent, documented first-pass
pipeline that does not require programming to operate.

# State of the field

A mature ecosystem of automatic eye-movement event-detection algorithms exists,
including adaptive velocity-based detection [@nystrom2010], noise-robust
fixation clustering (I2MC) [@hessels2017], and robust classification for dynamic
stimuli (REMoDNaV) [@dar2021]. These were developed for screens where the participant views a fixed two-dimensional display at
approximately constant depth — an assumption Kinarm experiments violate, since
the stimulus and the hand share a robotic workspace and the eye-to-stimulus
geometry must be modeled explicitly. Automation itself is a legitimate design
goal, but automated labels inherit the biases of the particular algorithm or
machine-learning technique used, and many such methods are never trained on
manually classified data under supervised learning; comparative evaluations
accordingly report substantial disagreement among algorithms and between
algorithms and human coders [@andersson2017]. Establishing reliable ground truth
is therefore a prerequisite in the Kinarm environment, whose ecological validity comes
at the cost of a more demanding geometry. The @singh2016 method was designed for
exactly this setting, but is published as an algorithm rather than as software.

GECKO is, to our knowledge, the first openly available tool that (i) reads the
native `.kinarm` format, (ii) implements the @singh2016 geometric
ocular-kinematics pipeline, and (iii) couples it to a frame-level,
human-in-the-loop labeling interface with full export control. We built a
dedicated tool rather than extending an existing package for two reasons. None
of the screen-based detectors read Kinarm data, account for the variable
eye-to-stimulus depth of a transverse workspace, or compare gaze kinematics
against the motion of the targets themselves — the comparison that separating
smooth pursuit from fixation requires when the stimulus is moving. In addition,
the laboratory workflow benefits from manual oversight that automated detectors
do not provide, a need recognized even within screen-based research, where
automatic detections are sometimes corrected by hand [@hessels2017].

# Software design

GECKO's architecture mirrors the hierarchical structure of Kinarm data itself:
files contain trials, and trials contain channels. The loaded-file object maps
onto trials, which map onto individual channels, and the GUI builds directly on
this object model without flattening or restructuring the underlying data. This
reduces the risk of cross-trial indexing errors and makes the codebase intuitive
to extend, compared with the alternative of loading everything into flat data
frames at startup. The processing pipeline — data loading, sentinel cleaning,
interpolation, and gaze-metric computation — is decoupled from the interface, so
the same modules can be imported into standalone scripts for batch reprocessing
and for sharing reproducible analysis code alongside publications. The pipeline
is implemented in Python and builds on NumPy [@harris2020] for the array
operations underlying the spherical-coordinate and angular-velocity
computations, SciPy [@virtanen2020] for signal filtering (the Butterworth
low-pass and Savitzky–Golay differentiation described below), and Matplotlib
[@hunter2007] for the interactive trial plots that underpin the labeling
interface.

All processing parameters are centralized in a single configuration file so that
labs can adapt the tool to their own setup without modifying the pipeline. These
include the sentinel threshold, the gap length below which interpolation happens
automatically, the assumed eye height above the stimulus plane, the foveal cone
angle used for FVR, and the filter settings: a 20 Hz fourth-order zero-phase
Butterworth low-pass applied to gaze before metric computation, and a
Savitzky–Golay filter used internally for the time derivatives in the
angular-velocity calculation. An accompanying technical reference documents every
transformation applied to the data, so that researchers can report their methods
accurately.

GECKO draws a deliberate line between what it automates and what it leaves to
the researcher. Preprocessing is automated: sentinel values are detected and
replaced, and small gaps (≤ 50 frames by default) are interpolated linearly
without prompting. Where the choice is scientifically consequential, control
returns to the user — larger gaps trigger an interactive preview offering linear
interpolation, saccadic (sigmoid) interpolation, or leaving the gap as `NaN`,
with each decision cached per trial and per channel so it is never silently
repeated. Gaze event labeling, by contrast, is entirely manual. GECKO presents
the computed kinematics and event markers (for example `TARGET_ON`) and lets the
user place, erase, and adjust event intervals frame by frame, but proposes no
labels of its own. The aim at this stage is to establish ground truth, so
transparency and correctability were judged more valuable than throughput.

Labeling a full dataset is a long process, so GECKO is built to be closed and
reopened without losing work. Trial quality marks, free-text notes, export
selections, label order, and per-file session state are all written to disk, so
reopening a `.kinarm` file restores the session exactly where it was left, and
trials can be labeled back-to-back without returning to the main window. Exports
are deliberately minimal. Per-frame gaze events are written as single-digit
integer codes (1 = saccade, 2 = pursuit, 3 = fixation, 9 = bad trial) in a CSV
file for human-readable inspection and in a compressed NPZ archive for storage
and downstream machine-learning ingestion. Channel data is kept as raw as the
researcher's interpolation choices dictate, so data-preparation decisions can be made at the
analysis stage.

# Research impact statement

GECKO was developed in the Sensorimotor Neuroscience and Learning Laboratory at
The Pennsylvania State University, where it is used to process gaze and kinematic
data from Kinarm reaching and interception experiments. Gaze analyses produced
with GECKO have supported work presented at the annual meeting of the Society
for the Neural Control of Movement [@muttee2026], comparing motor planning for
static reaches with planning for interception of moving targets. Beyond the
developing laboratory, GECKO has been adopted by two other groups that pair eye
tracking with Kinarm robotics: the Brain and Action Laboratory at the University
of Georgia (Dr. Deborah Barany), which studies goal-directed movement including
manual interception of moving objects, and the Sensorimotor Control and Robotic
Rehabilitation Research Laboratory at the University of Delaware (Dr. Jennifer
Semrau), which uses robotics to improve assessment and rehabilitation after
stroke. This adoption indicates a recurring need across any laboratory that uses
Kinarm and eye-tracking. The software is released under the MIT license with tagged
releases, a changelog, a contributing guide, and continuous-integration
workflows, and it ships with a worked demonstration dataset, a user manual, and
a technical reference documenting every processing step.


# AI usage disclosure

Generative AI (Anthropic's Claude, various model versions, mid-2025 to present)
was used as a programming assistant in the later stages of GECKO's development,
accelerating code implementation, debugging, and initial drafts of documentation
and manuscript text for the GUI application built on top of an initial
human-developed scientific pipeline and command-line prototype. It served as an
API reference and project-aware debugging aid, particularly for `numpy`, `scipy`,
and `matplotlib` usage. All design decisions, edge-case testing, and validation
were performed by the human developer, with algorithmically critical
implementations verified against @singh2016 and laboratory reference
implementations, and all AI-assisted outputs reviewed and approved by the 
authors of this manuscript.

# Acknowledgements

This work was supported by the National Science Foundation under Grant No.
2444649. The funder had no role in the design of the software, the analysis
approach, or the preparation of this manuscript. We thank Kinarm (BKIN
Technologies, Kingston, ON, Canada) for constructive feedback during the
development of GECKO.

# References
