# PointerCADv2-RL: контекст проекта

Актуально на 2026-07-28. Этот файл фиксирует текущее состояние проекта,
архитектурные решения и ограничения для последующей работы.

## Текущая задача

Репозиторий реализует Pointer-CAD v2 из статьи
`/Users/timofejbulgakov/Downloads/Pointer-CADv2.pdf`. Текущая цель —
добавить post-training на основе reinforcement learning. В качестве основного
референса по организации RL-этапа используется статья
`/Users/timofejbulgakov/Downloads/cadrille.pdf`.

Ближайшая практическая цель:

1. Подготовить воспроизводимый full-episode RL-датасет.
2. Сгенерировать несколько trajectories SFT-моделью для каждой задачи.
3. Исполнить trajectories, сохранить геометрию и raw metrics.
4. Построить preferred/rejected пары и реализовать full-episode DPO.
5. Сохранить совместимость формата данных с будущим online
   GRPO/PPO/CPPO-подобным обучением, чтобы не генерировать и не исполнять CAD
   повторно без необходимости.

## Архитектура PointerCAD v2

### Исходный датасет

Основной loader находится в `dataset/dataset.py`. Ожидаемая структура одного
CAD-моделя:

```text
<dataset_dir>/<chunk>/<model_id>/
├── prompt_abs.txt
├── prompt_exp.txt
├── plan/<model_id>_<part_id>.txt
├── parameter/<model_id>_<part_id>.pkl
├── vector/<model_id>_<part_id>.pkl
├── graph/<model_id>_<part_id>.bin
├── json/<model_id>_<part_id>.json
└── <model_id>.json
```

Split-файл содержит step-level идентификаторы вида
`<chunk>_<model_id>_<part_id>`. Исходный датасет и его
`train_val_test.json` остаются единственным источником истины. Производные RL
файлы не должны изменять или дублировать исходные CAD-данные без необходимости.

Перед первым запуском RL builders необходимо проверить эту ожидаемую структуру
на реальных примерах: один простой многошаговый sketch/extrude объект и один
объект с chamfer/fillet.

Исходное дерево может содержать symbolic links. При сохранении относительных
source paths нельзя вызывать `Path.resolve()`, иначе symlink будет раскрыт и
файл может оказаться физически вне настроенного `dataset_dir`. Episode builder
сохраняет логические пути через `Path.absolute().relative_to(...)`.

### SFT pipeline

- `train.py` реализует supervised fine-tuning.
- `models/processor.py` вставляет B-Rep embeddings и разворачивает CAD
  placeholders.
- `models/pointercad.py` содержит модель и autoregressive generation.
- `models/brep_embed.py` кодирует faces и edges текущего B-Rep.
- `models/parameter_encoder.py` кодирует length/angle dictionary, извлечённый
  из сгенерированного плана.
- `metrics/criterion.py` реализует SFT loss.

Модель генерирует одну CAD-операцию за вызов:

1. Qwen генерирует текстовый dimension-aware plan.
2. Из плана извлекается parameter map.
3. Label head генерирует типы structured CAD actions.
4. Parameter head выбирает length/angle из parameter map через cosine
   similarity.
5. Pointer head выбирает face/edge текущего B-Rep через cosine similarity.
6. Операция исполняется вне модели, после чего новый B-Rep становится входом
   следующего шага.

Таким образом, environment работает step-wise, даже если единицей DPO/RL
является полный episode.

SFT objective комбинирует четыре канала:

- plan/LM tokens;
- labels;
- parameters;
- B-Rep pointers.

Текущие веса в статье и `config/train.yaml`: `0.2 / 0.3 / 0.2 / 0.3`.

### Evaluation и CAD execution

- CAD-модель и операции находятся в `cadmodel/`.
- Полное progressive inference находится в `test.py`.
- Геометрические метрики находятся в `measurements/`.
- Legacy reward находится в `metrics/rewards.py`.

## Принятое решение по RL

### Единица обучения

Основной план — full-episode DPO, а не независимый step-level DPO.

Episode представляет полную попытку построить CAD-модель от начального
состояния до `model_end`, ошибки исполнения или лимита шагов. При этом
вероятность trajectory вычисляется пошагово:

```text
log pi(trajectory | prompt)
    = sum по шагам (
        plan token log-probabilities
        + label log-probabilities
        + conditional parameter log-probabilities
        + conditional pointer log-probabilities
      )
```

Каждый следующий шаг должен оцениваться на B-Rep, реально полученном после
исполнения предыдущего сохранённого действия. Нельзя представить full episode
как одну обычную текстовую completion без replay environment.

Step-level представление остаётся доступным как вложенная часть trajectory и
может использоваться для отладки и абляций.

### Организация производных данных

Все производные данные располагаются под:

```text
<source_dataset>/format/pointercad_rl/
```

Структура:

```text
format/pointercad_rl/
├── episodes/
│   ├── config.yaml
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
├── rollouts/
│   └── <run_id>/
│       ├── config.yaml
│       ├── trajectories/part-*.parquet
│       ├── steps/part-*.parquet
│       ├── scores/part-*.parquet
│       └── data/
│           ├── cad/
│           ├── mesh/
│           └── states/
└── preferences/
    └── <view_id>/
        ├── config.yaml
        ├── train.parquet
        └── validation.parquet
```

Использовать имя `data`, а не `artifacts`.

#### Episodes

`episodes` — лёгкий производный индекс, который группирует step-level split
по `(chunk, model_id)` и создаёт full-model задачи. Он не является новым
источником истины и может быть полностью пересоздан из исходного датасета.

Одна запись соответствует `(CAD model, prompt variant)` и содержит:

- `task_id`;
- split;
- prompt;
- ссылку на target CAD JSON;
- упорядоченный список исходных `part_id`;
- количество и типы операций;
- версию preprocessing.

#### Rollouts

Один rollout — одна полная сгенерированная trajectory.

- `trajectories` содержит summary episode, termination reason, validity,
  ссылки на финальный CAD/mesh, raw metrics и timings.
- `steps` содержит план, token IDs, parameter map, labels, parameters,
  pointers, behavior log-probabilities и результат исполнения каждого шага.
- `scores` кэширует log-probability одной trajectory под конкретным checkpoint,
  чтобы reference scores не вычислялись повторно для каждой пары.
- `data` содержит тяжёлые сгенерированные CAD, mesh и intermediate states.

Нужно хранить raw metrics, а не только scalar reward. Это позволяет менять
формулу reward без повторной генерации, CAD execution и вычисления уже
сохранённых метрик.

Offline DPO использует frozen rollout store. В будущем online RL использует тот
же формат как временный rollout buffer и сохраняет behavior/old-policy
log-probabilities.

#### Preferences

Preference view — дешёвое производное представление над rollout store. Оно
содержит только:

- `preferred_trajectory_id`;
- `rejected_trajectory_id`;
- их rewards и margin;
- reward version;
- pairing strategy.

Использовать термины `preferred/rejected`, а не `chosen/rejected`, во всех
dataset-facing схемах. Базовый DPO trainer реализован напрямую и не зависит от
текстового completion-контракта TRL.

Изменение reward или pairing rules должно требовать только перестроения
preference view. Rollouts при этом не генерируются заново.

## Реализованный RL data layer

Добавлены следующие файлы:

- `rl/schemas.py` — типизированные Arrow-friendly записи:
  `EpisodeRecord`, `TrajectoryRecord`, `StepRecord`, `ScoreRecord`,
  `PreferenceRecord`.
- `rl/storage.py` — атомарная запись и чтение Parquet/YAML, шардирование.
- `rl/episodes.py` — группировка исходного step-level split в full episodes.
- `rl/rollout_store.py` — append-only rollout store и каталог `data`.
- `rl/preference_view.py` — построение preferred/rejected пар.
- `rl/data.py` — read-only loaders и индексы.
- `rl/preference_data.py` — пересчёт reward из сохранённых raw metrics.
- `rl/cad_environment.py` — B-Rep graph, пошаговое CAD execution, экспорт и
  raw geometry metrics.
- `rl/rollout_generator.py` — воспроизводимая stochastic генерация full
  trajectories и запись через `RolloutStore`.
- `rl/prompts.py` — общий prompt для rollout generation и DPO replay.
- `rl/dpo_data.py` — join episode/preference/trajectory/step и DPO Dataset.
- `rl/likelihood.py` — joint likelihood по plan/label/parameter/pointer.
- `rl/dpo.py` — стандартный reference-anchored DPO objective.
- `dpo_train.py` — Accelerate training loop, validation и checkpoints.
- `preprocessing/build_rl_episode_index.py` — CLI episode builder.
- `preprocessing/generate_rl_rollouts.py` — CLI rollout generator.
- `preprocessing/build_rl_preferences.py` — CLI preference builder.
- `config/rl_dataset.yaml` — пример конфигурации episode builder.
- `config/rl_rollouts.yaml` — sampling, execution, metrics и output config.
- `config/rl_preferences.yaml` — пример reward/pairing конфигурации.
- `config/dpo_train.yaml` — конфигурация full-episode DPO.
- `scripts/rl_build_episodes.sh`, `scripts/rl_generate_rollouts.sh`,
  `scripts/rl_build_preferences.sh`, `scripts/dpo_train.sh` — launch scripts с
  активацией conda, GPU-настройками и логированием.
- `rl/README.md` — описание формата.

Динамические dictionaries сохраняются как canonical JSON strings; числовые
последовательности сохраняются как типизированные Arrow lists.

`models/pointercad.py::predict` опционально возвращает log-probabilities
фактически выбранных plan/label/parameter/pointer решений. Обычный интерфейс
сохранён; rollout generator включает расширение через
`return_log_probs=True`.

Rollout generator сейчас является однопроцессным baseline: sampling и
OpenCascade execution выполняются последовательно. Формат данных не зависит от
этого решения, поэтому worker isolation можно добавить после smoke-test
реального окружения.

## Что пока не реализовано

- Изолированный CAD executor для rollout workers.
- Precompute CLI для cached reference scores.
- Online GRPO/PPO/CPPO trainer.
- Финальная reward-функция и её тесты.

Устаревшие файлы `rl/trainer.py`, `rl/log_probs.py` и `rl/modeling.py`, которые
использовали старый интерфейс `values/pointers`, удалены. Их заменяют
`rl/dpo_data.py`, `rl/likelihood.py` и `rl/dpo.py`.

`rl_train.py` — legacy online-код, названный GRPO, но он также пока не является
целевой реализацией.

## Известные проблемы legacy RL и метрик

Не строить новое обучение непосредственно поверх текущих реализаций без
исправления следующих проблем:

- `rl_train.py` не сохраняет rollout log-probabilities old policy и не
  реализует настоящий clipped PPO ratio.
- В training group смешиваются argmax и stochastic candidates.
- Перезапуск iterator в `rl_train.py` не согласован с количеством элементов,
  потребляемых за iteration.
- Reference model периодически заменяется текущей policy, меняя смысл KL
  anchor.
- `metrics/rewards.py` утверждает, что последняя операция — `Extrude`, поэтому
  reward не покрывает chamfer/fillet.
- В aggregation vertex/edge/face reward есть ошибка приоритета условных
  операторов.
- Исключения CAD execution часто подавляются и превращаются в неразличимый
  нулевой reward.
- Текущий length reward зависит от ground-truth response length и может
  поощрять короткие, но геометрически неверные ответы.
- Текущие geometry accuracy metrics преимущественно recall-oriented, поэтому
  reward должен включать защиту от добавления лишней геометрии.

## Среда и проверки

По явной просьбе пользователя новый RL data code пока не запускался:

- builders не выполнялись;
- модули не импортировались;
- unit/smoke tests не запускались.

Пользователь сначала должен настроить среду. Для Parquet persistence нужен
`pyarrow`. Полный pipeline также требует PyTorch, DGL, Transformers, PEFT,
pythonocc-core/occwl и Accelerate. Custom DPO baseline не требует TRL.

До настройки среды не запускать RL builders или обучение без нового явного
запроса пользователя.

## Следующие шаги

1. Получить реальные примеры исходного датасета и проверить предположения
   `rl/episodes.py` о split IDs, prompt paths и final target JSON.
2. После настройки среды запустить episode builder на небольшом subset и
   проверить Parquet schema.
3. Запустить rollout generator на нескольких задачах и проверить CAD states,
   raw metrics и behavior log-probabilities.
4. Построить preference view и проверить joint log-probability на одном
   известном episode.
5. Запустить DPO smoke test, затем небольшой training run.
6. Добавить CAD worker isolation и cached reference-score builder.
7. После стабилизации DPO реализовать online RL поверх того же episode/rollout
   формата.

## Правила работы с репозиторием

- Worktree содержит многочисленные изменения пользователя, включая untracked
  RL-файлы. Не удалять и не откатывать несвязанные изменения.
- Не изменять исходный CAD-датасет; создавать только производные файлы под
  `format/pointercad_rl`.
- Сохранять dataset-facing имена `preferred/rejected` и каталог `data`.
- Версионировать preprocessing, metrics, reward, checkpoints и sampling config,
  чтобы derived data можно было воспроизвести.
- Не добавлять text-only completion adapters: PointerCAD DPO должен сохранять
  structured full-episode semantics.
