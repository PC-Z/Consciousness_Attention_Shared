# Peterka et al. 2026: Global Context in V1

## Citation

Peterka DS, Imai F, Ross JM, Bastos G, Hornick M, Rachmany L, Gallimore CG,
Hockley A, Hamm JP. *Global context rapidly shapes sensory responses in V1*.
bioRxiv. 2026. DOI:
[10.64898/2026.01.07.698143](https://doi.org/10.64898/2026.01.07.698143).
Full text: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.07.698143v1),
[PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803279/).

This is the Peterka et al. preprint cited on slide 3 of the oddball experiment
proposal. It is the direct source for both the rapid AAAAB training effect and the
P5-A/B/C test predictions.

## Figure S1: Exact Train Analysis

- The initial training contained 35 AAAAB or reversed BBBBA sequence repeats.
- Training and random-control trials were divided into five bins of seven trials.
- Figure S1A shows the first seven and last seven repeats across sequence positions
  P1-P5. The initially enhanced P5-B response was significant in the first bin
  (`p=.009`) and absent in the final bin (`p=.85`). Responses to repeated A remained
  below the random-sequence control at several positions.
- The caption reports 152 responsive neurons from five mice, with four runs per
  mouse. Neuron-level linear mixed-effects models included mouse as a random effect.
- Locomotion trials were not excluded from Figure S1 so every mouse retained enough
  trials in every bin. Figure S1B reports no bin or bin-by-context locomotion effect.

Slide-ready conclusion: **V1 learned the global AAAAB structure within fewer than
10 repeats: the initial P5-B enhancement disappeared after approximately seven
sequences, while adaptation to repeated A persisted.**

## Figures 1D and 2: P5 Context Effects

The paper compared each P5 event with the same orientation in a random-sequence
control. That orientation-matched control is essential to the paper's definition of
deviance detection.

| P5 event | Context | Result versus same-orientation random control |
|---|---|---|
| A in AAAAA | globally deviant, locally repeated | suppressed: `t(92)=-4.06`, `p=7.24e-5` |
| B in AAAAB | locally deviant, globally predictable | no enhancement: `t(86)=0.015`, `p=.988` |
| C in AAAAC | locally and globally deviant | enhanced: `t(119)=2.89`, `p=.004` |

Figure 2 reports C deviance-detecting responses in about 10.2% of all active
pyramidal neurons or 18.3% of C-responsive neurons. The corresponding proportions
for A and B were about 3.3%. The paper used deconvolved, standardized estimated
firing rates from layer 2/3 excitatory V1 neurons; these percentages are not direct
threshold targets for the current whole-brain denoised fluorescence matrix.

Slide-ready conclusion: **At P5, gain reflected both local adaptation and global
predictability: repeated A was suppressed, predictable B matched control, and novel
C was enhanced.**

## Notebook 04 Mapping

Notebook 04 now uses a single fixed `VISp` population for train and test. Test A/B/C
trial counts are balanced, and matched heatmaps retain the same cells, row scaling,
row order, and color limits.

The current experiment lacks the random-sequence control, so it cannot reproduce
the paper's formal deviant-minus-control statistic. It instead reports three named
proxies:

| Paper prediction | Current proxy | Remaining confound |
|---|---|---|
| P5-A suppressed | P5-A versus P4-A within AAAAA | sequence position and repetition |
| predictable P5-B not enhanced | test P5-B versus late-train P5-B | phase and elapsed time |
| P5-C enhanced | balanced test P5-C versus P5-B | orientation identity |

Train uses the paper's first-seven versus last-seven comparison and adds a
parameterized full trajectory because the current run contains 100 sequences. A
two-one-sided equivalence test is used for predictable B; a non-significant ordinary
difference is not interpreted as proof of equivalence.

All neuron-level tests describe one recording. A multi-mouse mixed-effects model and
the random-sequence control are required before claiming replication.

## 组会汇报表述

- **训练期（Figure S1）**：P5-B 的初始增强主要出现在前约 7 个 AAAAB 序列，随后快速衰减并在训练末期消失，提示 V1 在少于 10 次序列重复内形成了对 P5-B 的全局预测；重复 A 的刺激特异性适应仍然存在。
- **测试期（Figures 1D、2）**：相对于相同取向的随机序列对照，AAAAA 的 P5-A 受到抑制，AAAAB 中全局可预测的 P5-B 不再增强，而同时违反局部和全局规则的 AAAAC P5-C 显著增强。
- **本预实验的边界**：当前可以检查 train P5-B 的早晚变化以及固定 V1 神经元的平衡 A/B/C 响应，但没有随机序列对照，因此不能把代理对比直接命名为论文意义上的 deviance detection。

## Related Background

- Hamm JP, Yuste R. *Somatostatin Interneurons Control a Key Component of
  Mismatch Negativity in Mouse Visual Cortex*. Cell Reports. 2016.
  [DOI 10.1016/j.celrep.2016.06.037](https://doi.org/10.1016/j.celrep.2016.06.037).
- Hamm JP et al. *Cortical ensembles selective for context*. PNAS. 2021.
  [DOI 10.1073/pnas.2026179118](https://doi.org/10.1073/pnas.2026179118).
