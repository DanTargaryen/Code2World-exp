# CoinRun 游戏框架 — config schema 设计

> 目标:把 procgen CoinRun(`BasicAbstractGame` 派生)的硬编码机制系统性提成
> **float/int/bool options**,让一份可读的 YAML config 就能定义游戏机制(LLM 只写
> config)。config 既驱动数据生成(procgen options,无需重编译),又当 code condition
> 喂 text encoder。默认值全部 = 原版 coinrun.cpp,空 config 复刻 vanilla。

## 设计原则

1. **框架底座 = procgen `BasicAbstractGame`**(17 游戏共用,已提供 entity/collision/
   grid/spawn/physics/render 全套)。CoinRun 只重写少数虚函数定义机制 —— 我们把这些
   机制的硬编码常量提成 options。
2. **默认 = 原值**:每个 option 默认值等于 coinrun.cpp 里的硬编码,零漂移(已由
   `verify_config.py` 的 md5 一致性保证)。
3. **分组**:config 按机制语义分组(physics / reward / termination / difficulty /
   hazards / terrain),可读性优先。
4. **粒度**:先覆盖 CoinRun 全部可调机制;后续横向推到 maze/climber 时,通用机制
   (movement/collision/reward)沉到共享 options,游戏专属的单列。

## config schema(目标全量)

```yaml
name: <str>              # 变体名(bookkeeping)
game: coinrun            # 目前仅 coinrun

physics:                 # 已实现 ✅
  gravity: 0.2           # 下落加速度/步
  max_jump: 1.5          # 跳跃冲量 / vy 上限
  max_speed: 0.5         # 水平最大速度
  air_control: 0.15      # 空中水平控制比例
  mixrate: 0.2           # 地面速度混合率

reward:                  # 部分实现
  goal_reward: 10.0      # ✅ 到达金币奖励
  # timeout_penalty / step_penalty 可加(原版无,属新增机制)

termination:             # 待实现 —— 致死规则(handle_*_collision)
  die_on_enemy: true     # 碰敌人是否 done(原版 true)
  die_on_saw: true       # 碰锯齿是否 done(原版 true)
  die_on_lava: true      # 碰岩浆是否 done(原版 true)

difficulty:              # 待实现 —— generate_coin_to_the_right 布局难度
  max_difficulty: 3      # dif = randn(max_difficulty)+1,控制段数/障碍密度
  # num_sections 由 dif 派生,一般不单独暴露

hazards:                 # 待实现 —— 危险物开关+密度(现挤在 debug_mode 位里)
  allow_pit: true        # 是否生成坑(debug_mode bit1)
  allow_crate: true      # 是否生成木箱(debug_mode bit2)
  allow_monsters: true   # 是否生成移动敌人
  saw_spawn_rate: 0.2    # 锯齿生成概率相关(randn(10) < 2*dif 的系数)
  enemy_spawn_rate: 0.1  # 敌人生成概率相关(randn(10) < dif 的系数)

terrain:                 # 待实现 —— 地形起伏
  allow_dy: true         # 是否允许高度变化(debug_mode bit3);false=平地
  max_dy_range: 4        # dy = randn(4)+1+dif/3 的范围
```

## 落地分期

| 期 | 内容 | 状态 |
|---|---|---|
| P0 | physics(5)+ goal_reward | ✅ 已实现(`feat/coinrun-config-options`) |
| P1 | termination 致死规则(die_on_enemy/saw/lava) | 待做 |
| P2 | hazards 开关(allow_pit/crate/monsters,替代 debug_mode 位) | 待做 |
| P3 | difficulty + hazards 生成率 + terrain | 待做 |
| P4 | 横向:抽通用机制,推到 maze/climber(多游戏) | 远期 |

## 实现要点(每期相同套路)

1. **game.h** `GameOptions` 加 `coinrun_*` 字段(默认=原值)。
2. **game.cpp** `parse_options` 加 `consume_{float,int,bool}`。
3. **coinrun.cpp** 硬编码常量改读 `options.coinrun_*`。
4. **game_config.py** `MECHANIC_TO_OPTION` + `VANILLA` 加映射;支持分组 YAML。
5. **env.py** `coinrun_config` 透传(已支持任意 `coinrun_*` key)。
6. **verify_config.py** 加该机制的 counterfactual 自检 + 复跑零漂移。
7. pod 重编译 + 自检。

## 注意

- `allow_pit/crate/dy` 原版用 `debug_mode` 的位开关(bit1/2/3)。提成独立 bool option
  更可读,但要保证与 debug_mode 位不冲突(二选一,建议新 option 优先、debug_mode 保留兼容)。
- 生成率类(saw/enemy_spawn_rate)原版是 `randn(10) < 2*dif` 这类整数比较,提成 float
  概率要改判定式(如 `rand_gen.rand01() < rate`),需重验零漂移的等价性(默认值要让
  概率等于原判定)。这是 P3 最需要小心的点。
```
