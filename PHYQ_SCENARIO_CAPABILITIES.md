# Phy-Q Scenario Capabilities

Reference table mapping each Phy-Q physical reasoning scenario to the actions, processes, and abilities required to solve its levels.

**Legend:** ✓ = required · ◐ = sometimes · — = not needed · Rows = scenarios · Columns = abilities (see legends below).

## Capability Matrix

| Scenario | SB | SM | TP | GC | PC | BI | DP | IP | RO | SL | FA | BO | WT | HT | WD | SH | MS | SO | SW | CL | TM | BT |
|----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **SF** single_force | ✓ | — | ✓ | ✓ | ✓ | ◐ | ✓ | — | — | — | — | ◐ | — | — | — | — | — | — | — | — | — | — |
| **MF** multiple_forces | ✓ | ✓ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ | — | — | — | ◐ | — | — | — | — | ✓ | — | — | — | — | — |
| **RL** rolling | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | ◐ | ◐ | — | — | — | ◐ | ◐ | — | — | — | — | — |
| **FL** falling | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ◐ | — | ✓ | ◐ | — | ◐ | — | — | ◐ | — | — | — | — | — |
| **SL** sliding | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ◐ | ◐ | — | — | — | ◐ | ◐ | — | — | — | — | — |
| **BN** bouncing | ✓ | ◐ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ | — | — | — | ✓ | — | — | ◐ | — | ◐ | — | — | — | — | — |
| **RW** relative_weight | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ◐ | ◐ | ◐ | ✓ | — | — | ◐ | ✓ | — | — | — | — | — |
| **HT** relative_height | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ◐ | — | ✓ | ◐ | ◐ | ✓ | ◐ | ◐ | ✓ | — | — | — | — | — |
| **WD** relative_width | ✓ | ◐ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ | — | — | — | ◐ | — | ◐ | ✓ | — | ◐ | — | — | — | — | — |
| **SD** shape_difference | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ◐ | ✓ | ◐ | — | ◐ | — | ✓ | ✓ | — | — | — | — | — |
| **NG** non_greedy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ◐ | ✓ | ✓ | ◐ | ✓ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | ✓ | ◐ | ◐ | ◐ | — |
| **SA** structural_analysis | ✓ | ◐ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ◐ | ◐ | ✓ | ◐ | — | ◐ | — | ◐ | ✓ | — | ✓ | — | — | — |
| **CP** clearing_paths | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | ◐ | ◐ | ✓ | ◐ | — |
| **AT** adequate_timing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ◐ | ✓ | ◐ | — | — | — | — | ✓ | ◐ | ◐ | ◐ | ✓ | — |
| **MN** manoeuvring | ✓ | ◐ | ✓ | ✓ | ✓ | ◐ | ✓ | ◐ | — | — | — | ◐ | — | — | ◐ | — | ✓ | — | — | — | — | ✓ |

### Ability columns

| | | | |
|--|--|--|--|
| **SB** Shoot bird | **SM** Shoot multiple birds | **TP** Trajectory | **GC** Ground collision |
| **PC** Platform collision | **BI** Block impact | **DP** Direct pig kill | **IP** Indirect pig kill |
| **RO** Rolling | **SL** Sliding | **FA** Falling / gravity | **BO** Bouncing |
| **WT** Weight reasoning | **HT** Height reasoning | **WD** Width / path | **SH** Shape reasoning |
| **MS** Multi-step sequencing | **SO** Strategic ordering | **SW** Structural weak-point | **CL** Path clearing |
| **TM** Timed actions | **BT** Bird tap / special ability | | |

## Per-Scenario Breakdown

### Category 1 — Force

| Scenario | Description | Required abilities |
|----------|-------------|-------------------|
| **single_force** | Destroy pig with a single force | Shoot bird → plan trajectory → collide with ground/platform/blocks → direct pig kill |
| **multiple_forces** | Destroy pig(s) with multiple forces | All of single_force **+** shoot multiple birds **+** multi-shot sequencing |

### Category 2 — Motion

| Scenario | Description | Required abilities |
|----------|-------------|-------------------|
| **rolling** | Roll circular object onto pig | Shoot bird into circular block → block rolls along ground/platform → indirect pig kill |
| **falling** | Drop objects onto pig | Shoot to dislodge/topple block → falling/gravity → indirect pig kill |
| **sliding** | Slide non-circular object to pig | Shoot into non-circular block → sliding motion on surface → indirect pig kill |
| **bouncing** | Bounce bird off surface to reach pig | Shoot bird → bounce off platform/surface → reach unreachable pig (reflection angle) |

### Category 3 — Complex Reasoning

| Scenario | Description | Required abilities |
|----------|-------------|-------------------|
| **relative_weight** | Move object with correct weight | Compare object weights → choose movable object → roll/slide/fall → indirect kill |
| **relative_height** | Move object with correct height | Compare object heights → topple correct block → fall through gap → indirect kill |
| **relative_width** | Select correct opening/width | Compare opening widths → choose correct path/entrance → trajectory through gap → pig kill |
| **shape_difference** | Move/destroy object with correct shape | Compare shapes (rollable vs not) → destroy correct support → roll/fall chain → indirect kill |
| **non_greedy** | Select actions in correct order | Multi-shot sequencing **+** strategic ordering (avoid destroying wrong pigs first) **+** consequence prediction |
| **structural_analysis** | Break stability at weak point | Identify structural weak point → shot at key block → cascade collapse → pig kill |
| **clearing_paths** | Create path before reaching target | Step 1: reposition/clear block → Step 2: roll/move object through opened path → indirect kill |
| **adequate_timing** | Perform actions within time constraints | Multi-shot **+** roll objects to trigger **+** destroy prop at precise moment **+** timed second action |
| **manoeuvring** | Activate bird power correctly | Shoot blue bird → tap during flight to split → manoeuvre split birds to multiple targets |

## Ability Stack by Complexity

```
Tier 1 (basic — scenarios 1–2)
  shoot bird → trajectory → ground/platform collision → direct kill

Tier 2 (+ motion — scenarios 3–6)
  + block impact → indirect kill → rolling | sliding | falling | bouncing

Tier 3 (+ property reasoning — scenarios 7–10)
  + compare weight | height | width | shape → choose correct object/path

Tier 4 (+ planning — scenarios 11–13)
  + multi-step sequencing → strategic ordering → path clearing → structural analysis

Tier 5 (+ temporal/ability — scenarios 14–15)
  + timed actions → bird tap / special ability activation
```

## PDDL Agent Support (Strike-Eagle)

| Ability | PDDL support |
|---------|--------------|
| Shoot bird, trajectory, ground/platform collision | Strong |
| Shoot multiple birds, multi-step sequencing | Partial |
| Block impact, indirect kill | Partial |
| Rolling / sliding / falling physics | Partial (KB models exist) |
| Bouncing / reflection | Partial |
| Weight / height / width / shape reasoning | Weak |
| Non-greedy ordering, structural analysis | Weak |
| Path clearing, timed actions | Weak |
| Bird tap / manoeuvring | Minimal (predicates exist, logic mostly commented out) |

## References

- Xue et al. "Phy-Q as a measure for physical reasoning intelligence" — [Nature Machine Intelligence, 2022](https://www.nature.com/articles/s42256-022-00583-4)
- Benchmark README: `external/phy-q/README.md`
- Scenario metadata: `ScienceBirds/win6.6/win/Levels/phy_q/manifest.json`
- Metrics module: `agents/pddl/phyq_metrics.py`
