# Visual Detector｜ローカル構築・操作手順

指定した主画面の領域を取得し、Liquid AI d1-2BのSystem One APIで領域検出・全体判定・全体推論を行います。画像と推論はこのPC内で処理します。

## 1. 準備

Python、Tkinter、画面取得が使えるデスクトップ環境を用意します。GPUを使う場合は、PyTorchからCUDA、Intel XPU（Core Ultraの内蔵Arc GPUなど）、またはApple Silicon MPSが利用できる必要があります。Intel NPUはOpenVINO経由の別バックエンドです（[6章](#6-intel-gpuxpu版)・[7章](#7-intel-npu版)）。CPU実行も選べます。

公開リポジトリにはモデル重みと仮想環境を含めません。次の手順で環境とモデルを用意してください。

**ディスクの空き容量**：モデル3.8 GB、仮想環境1〜4 GB（XPU版は約4 GB）、NPU用の変換モデル約4 GBが必要です。取得途中で容量が尽きると`os error 112`で失敗するため、事前に15 GB以上空けてください。

### Windows / PowerShell

リポジトリを取得したフォルダー（`app.py`がある場所）で実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# GPUを使う場合、PyTorch公式の環境に合ったCUDA版インストールコマンドを先に実行
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe tools\diagnose.py
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# GPUを使う場合、PyTorch公式の環境に合ったインストールコマンドを先に実行
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python tools/diagnose.py
```

GPUを選ぶ場合、診断結果の`accelerators.cuda`・`accelerators.xpu`・`accelerators.mps`のいずれかが`true`で、`accelerators.selected_gpu`に使うデバイスが表示されることを確認します。PyTorchの導入方法は[公式手順](https://pytorch.org/get-started/locally/)を参照してください。

## 2. モデルの配置

このフォルダーに`models/d1-2B/model.safetensors`と`model.lock.json`がある場合、取得済みです。別のPCでは公式モデルのコードを確認したうえで次を実行します。

```powershell
.\.venv\Scripts\python.exe tools\download_model.py --accept-model-code
```

macOS/Linuxは`.venv/bin/python tools/download_model.py --accept-model-code`です。モデルの取得には通信が必要です。推論時はローカルのモデルだけを読み込みます。

## 3. 起動とGPU／CPUの選択

Windowsでは`run_local.bat`、macOS/Linuxでは`bash run_local.sh`、または仮想環境のPythonで`python app.py`を実行します。

小窓右上の「実行」で**GPU**・**NPU**・**CPU**を選びます。停止中に変更すると、次に「開始」を押したときに選んだデバイスでモデルを読み込みます。読込後は領域表示の下に「実行先：d1-2B / xpu (Intel(R) Arc(TM) 130V GPU (8GB)) / bfloat16 / …」のように、実際の実行先・精度が表示されます。GPU・NPUが利用できない場合は理由を表示して選択を戻すので、CPUを選択してください。CPU実行はfloat32です。GPU実行は対応する精度を自動で選びます（CUDA・XPUはbfloat16、非対応ならfloat16）。CLIでは起動時の指定として`--device cuda`、`--device xpu`、`--device mps`、`--device npu`、`--device cpu`も使えます。

選んだGPU・NPUで実際に動かない場合（モデルの一部がCPUに残った、推論がCPUで実行された、OpenVINOがNPU以外で実行しようとした）はエラーとして停止し、CPUへ自動で切り替えて成功扱いにはしません。

XPU版・NPU版は別の仮想環境（`.venv-xpu`）や追加の変換手順が必要です。[6章](#6-intel-gpuxpu版)・[7章](#7-intel-npu版)を参照してください。

1. 「領域選択」を押し、主画面で監視したい範囲をドラッグします。
2. 小窓を選択した範囲の外へ移動します。
3. モードと指示を設定して「開始」を押します。
4. モデルの読込後、選択領域の取得と推論が繰り返されます。「停止」で新たな取得を止めます。

小窓が選択範囲に重なると画面取得を一時停止します。画面構成を変えた場合は領域を選び直してください。

## 4. 3つのモード

「全体」は選択した画面領域の画像全体を指します。

| モード | 操作 |
|---|---|
| 領域検出 | 対象指示に従ってBBoxを表示します。初期例は「映っている魚をすべて囲む」。信頼度下限は0.00〜1.00です。 |
| 全体判定 | 判定指示の初期例は「魚が映っているか？」。判定しきい値の初期値は0.20で、スコアが超えたフレームを陽性とします。陽性フレームの直近3枚を取得時刻付きで表示します。位置の指示の初期例は「映っている魚をすべて囲む」。陽性判定後だけ、別の推論でその位置を探します。位置が不要なら空欄にできます。 |
| 全体推論 | 「候補を編集」で分類候補を1行ずつ入力します。初期値は「魚がある」のみ。「該当しない」が自動で加わり、最高スコアの1件を表示します。 |

指示・しきい値・候補・実行デバイスは停止中に変更します。更新間隔は0.5 / 1.0 / 2.0 / 5.0 / 10.0秒から選べます。これは次の取得を始める最短間隔であり、推論が遅い場合に画像をためません。CPUで1フレームに約10秒かかる場合（Core Ultra 5 236V・メモリ16 GBでの実測は8〜17秒）、0.5秒を選んでも結果は0.5秒ごとになりません。取得頻度を抑えたい場合は10.0秒を選びます。「推論」は画像の縮小からモデル応答まで、「処理合計」は画面取得開始から表示までの時間です。

## 5. 動作確認

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
.\.venv\Scripts\python.exe -m tests.real_modes_smoke path\to\cats.jpg
.\.venv\Scripts\python.exe -m tests.gui_modes_smoke
```

静止画を個別に試す場合は`tools/check_model.py`を使います。GPU／CPUや環境の状態は`tools/diagnose.py`で確認できます。実モデルの確認には手元の画像を指定してください。画像ファイルはリポジトリに含まれません。

処理時間・メモリ・GPU/NPU使用率・ページングをまとめて測る場合は`tools/benchmark.py`を使います（Windows）。実モデルでGUIの停止・再開・デバイス切り替えを確かめる場合は`tests/real_gui_device_switch.py`を使います。

```powershell
.\.venv\Scripts\python.exe tools\benchmark.py path\to\cats.jpg --device cpu --runs 3 --output docs\benchmarks\cpu.json
.\.venv-xpu\Scripts\python.exe -m tests.real_gui_device_switch path\to\cats.jpg --accelerator GPU
```

`diagnose.py`の出力にはモデルの保存先（ユーザー名を含むパス）が出ます。公開リポジトリへ貼る場合は削ってください。`benchmark.py`の出力にはパス・ユーザー名・画像内容を含めません。

## 6. Intel GPU（XPU）版

Intel Core Ultra（例：Core Ultra 200V）の内蔵Arc GPUで、PyTorchのXPUバックエンドを使います。CPU版の`.venv`とは別の仮想環境にします。

1. Intel GPUドライバーを最新にします（[Intel Arc & Iris Xe Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)、またはIntel Driver & Support Assistant）。
2. [PyTorch XPU版](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)を別の仮想環境に入れます。

```powershell
python -m venv .venv-xpu
.\.venv-xpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-xpu\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
.\.venv-xpu\Scripts\python.exe -m pip install -r requirements-xpu.txt
.\.venv-xpu\Scripts\python.exe -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
.\.venv-xpu\Scripts\python.exe tools\diagnose.py
```

`True`とGPU名（例：`Intel(R) Arc(TM) 130V GPU (8GB)`）が表示され、診断結果で`accelerators.xpu`が`true`、`selected_gpu`が`xpu`になれば準備完了です。XPU版のwheelは依存ランタイム（oneAPI）を含むため約4 GBあり、取得に時間がかかります。`requirements-xpu.txt`は`accelerate`を追加し、重みをCPUへ全量展開せずXPUへ直接読み込みます（内蔵GPUはシステムメモリを共有するため、二重に持つとメモリ不足になります）。

3. 起動は`.\.venv-xpu\Scripts\python.exe app.py`（または`--device xpu`を付ける）です。「実行」のGPUがXPUを指します。モデルはこのフォルダーの`models/d1-2B`を共有するので、再取得は不要です。

初回の推論はGPUカーネルのコンパイルで10秒前後かかります。2回目以降は1フレーム0.3〜0.7秒程度です（計測値は[検証結果](INTEL_XPU_NPU_REPORT_JA.md)）。

メモリ16 GB機では、XPU版の推論中にプロセスが約8.5 GB（うちGPU共有約5 GB）を使います。他のアプリで多くのメモリを使っているとページングが起きるため、ブラウザーなどを閉じて使ってください。

## 7. Intel NPU版

Intel Core Ultra のNPU（Intel AI Boost）で、OpenVINOを使って推論する別バックエンドです。d1-2Bの画像エンコーダーと言語モデルをOpenVINO形式に変換し、NPU向けにコンパイルして使います。トークン化や回答の読み取りはd1自身のコードをそのまま使うので、3モードの出力はCPU・GPU版と同じ形式です。

1. Intel NPUドライバーを最新にします（[Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)）。デバイスマネージャーに「Intel(R) AI Boost」が表示されることを確認します。
2. CPU版の`.venv`に[OpenVINO](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)を追加し、`NPU`が見えることを確認します。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-npu.txt
.\.venv\Scripts\python.exe -c "import openvino as ov; c=ov.Core(); print(c.available_devices, c.get_property('NPU','FULL_DEVICE_NAME'))"
```

`['CPU', 'GPU', 'NPU'] Intel(R) AI Boost`のように表示されれば準備完了です。

3. モデルをNPU用に変換します。初回のみ必要で、`models/d1-2B-npu/`（約3.7 GB）に保存されます。

```powershell
.\.venv\Scripts\python.exe tools\export_npu.py
```

変換は「PyTorchでの基準計算 → OpenVINO形式へ変換 → OpenVINO CPUで基準と照合 → NPUコンパイル」の順に進み、各段階の時間とメモリを表示します。変換中は一時的に約17 GBのメモリ（ページファイル込み）を使い、16 GB機で約7分かかります。NPUコンパイルの結果は`.ov_cache/`に保存され、2回目以降のアプリ起動ではモデル読込が約15秒になります。d1-2Bを取得し直した場合（revisionが変わった場合）は再変換が必要です。

4. アプリの「実行」で**NPU**を選ぶか、`--device npu`で起動します。読込後、領域表示の右に「実行先 npu・fp16」と表示されます。

NPU版の制約：

- NPUは固定形状が必要なため、画像を640×480に変形して入力します。BBoxは画像に対する比率なので、表示位置は正しく戻ります。
- 指示・分類候補を含めた入力の長さに上限（640トークン）があります。通常の指示や、短い候補15件までは収まります。超えた場合はエラーを表示して停止します。長い候補を使う場合は`tools\export_npu.py --seq-len 1024`で変換し直します（推論は遅くなります）。
- NPUで実行できない場合（ドライバー・変換モデルがない、OpenVINOがNPU以外で実行しようとした）は、理由を表示して停止します。CPUへ自動で切り替えません。

変換で対処した問題（NPUが扱えない演算と、その置き換え）と計測結果は[検証結果](INTEL_XPU_NPU_REPORT_JA.md)にあります。

## 参考資料

- [Liquid AI d1-2B モデルカード](https://huggingface.co/LiquidAI/d1-2B)
- [PyTorch インストール手順](https://pytorch.org/get-started/locally/)
- [PyTorch Intel GPU（XPU）導入手順](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)
- [OpenVINO NPU デバイス](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO PyTorchモデルの変換](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-pytorch.html)
- [Pillow ImageGrab](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html)
