# Task 3 ACT Chunking Robustness Notes

Key evidence for the report:

- D zero-shot action error improves from B-only to ABC-style training. The best current model is `act_ABC_aug` with Action L1 `0.2134`.
- The visual shift proxy is lower for ABC than B when comparing RGB mean to D: B mean RGB L2 `0.2948`, ABC mean RGB L2 `0.0353`.
- All models are strongly smoother than D ground-truth actions. The ground-truth D mean step delta is about `0.1738`, while predicted step deltas are around `0.0162` to `0.0215`.
- This supports an ACT chunking interpretation: chunking stabilizes predictions under visual shift, but also produces over-smoothed action streams.
- The main tradeoff is accuracy versus chunk-boundary stability. `act_ABC_aug` has the best D Action L1, but its boundary amplification is `1.90x` the D ground-truth boundary jump.
- The largest boundary amplification is `act_ABC_aug` at `1.90x`.
- The most over-smoothed model by GT-minus-pred step delta is `act_B`.

Suggested report wording:

Under D visual shift, ACT's action chunking acts as a stabilizer: predicted within-chunk step changes are far smaller than the D demonstration action changes, so the policy does not jitter frame-by-frame. However, the chunk queue also delays correction and concentrates discontinuities at chunk refresh boundaries. Multi-environment training improves D action accuracy, especially gripper prediction, but the larger and more diverse training distribution can increase chunk-boundary jumps. Therefore the robustness is not simply 'more diversity is smoother'; it is an accuracy-stability tradeoff induced by chunked open-loop execution.
