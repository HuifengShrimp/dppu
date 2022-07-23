# CHANGELOGS

> Instrument:
>
> - Add `[Feature]` prefix for new feature
> - Add `[Bugfix]` prefix for bugfix
> - Add `[API]` prefix for API change

## staging
> please add your unrelease change here.

## 20200308
- [PPU] 0.0.4 release
- [Feature] add silent ot support for various ot scenarios (chosen/correlated/random messages, chosen/correlated/random choices, 1o2/1oN)
- [Feature] add non-linear computation protocols based on silent ot (comparison, truncation, b2a, triple, randbit, etc)
- [Feature] add a 2PC protocol: Cheetah
- [Improvement] concatenate is a lot faster
- [API] add RuntimeConfig.enable_op_time_profile to ask PPU to collect timing profiling data
- [Bugfix] fixed pphlo.ConstOp may lose signed bit after encoding

## 20220303
- [ppu] 0.0.3 release
- [API] merge (config.proto, executable.proto, types.proto) into single ppu.proto.
- [API] change RuntimeConfig.enable_protocol_trace to enable_action_trace.
- [API] change RuntimeConfig.fxp_recirptocal_goldschmdit_iters to fxp_reciprocal_goldschmdit_iters.
- [API] add RuntimConfig.reveal_secret_condition to allow reveal secret control flow condition.
- [Bugfix] Fixed SEGV when reconstruct from an ABY3 scalar secret
- [Feature] Left/right shift now properly supports non-scalar inputs

## 20220217
- [ppu] 0.0.2.3 release

## 20220211
- [ppu] 0.0.2.2 release
- [Bugfix] Fix exception when kkrt psi input is too small

## 20220210
- [ppu] 0.0.2.1 release
- [Bugfix] Fix exception when streaming psi output directory already exists 

## 20220209
- [ppu] 0.0.2 release.
- [Feature] Support multi-parties psi

## 20210930
- [ppu] Init release.
