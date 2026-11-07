# Visual Detector｜ローカル構築・操作手順

指定した主画面の領域を取得し、Liquid AI d1-2BのSystem One APIで領域検出・全体判定・全体推論を行います。画像と推論はこのPC内で処理します。

## 1. 準備

Python、Tkinter、画面取得が使えるデスクトップ環境を用意します。GPUを使う場合は、PyTorchからCUDAまたはApple Silicon MPSが利用できる必要があります。CPU実行も選べます。

公開リポジトリにはモデル重みと仮想環境を含めません。次の手順で環境とモデルを用意してください。

### Windows / PowerShell

```powershell
cd liquid_desktop_mvp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# GPUを使う場合、PyTorch公式の環境に合ったCUDA版インストールコマンドを先に実行
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe tools\diagnose.py
```

### macOS / Linux

```bash
cd liquid_desktop_mvp
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# GPUを使う場合、PyTorch公式の環境に合ったインストールコマンドを先に実行
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python tools/diagnose.py
```

GPUを選ぶ場合、診断結果の`accelerators.cuda`または`accelerators.mps`が`true`であることを確認します。PyTorchの導入方法は[公式手順](https://pytorch.org/get-started/locally/)を参照してください。

## 2. モデルの配置

このフォルダーに`models/d1-2B/model.safetensors`と`model.lock.json`がある場合、取得済みです。別のPCでは公式モデルのコードを確認したうえで次を実行します。

```powershell
.\.venv\Scripts\python.exe tools\download_model.py --accept-model-code
```

macOS/Linuxは`.venv/bin/python tools/download_model.py --accept-model-code`です。モデルの取得には通信が必要です。推論時はローカルのモデルだけを読み込みます。

## 3. 起動とGPU／CPUの選択

Windowsでは`run_local.bat`、macOS/Linuxでは`bash run_local.sh`、または仮想環境のPythonで`python app.py`を実行します。

小窓右上の「実行」で**GPU**または**CPU**を選びます。停止中に変更すると、次に「開始」を押したときに選んだデバイスでモデルを読み込みます。GPUが利用できない場合はCPUを選択してください。CPU実行はfloat32です。GPU実行は対応する精度を自動で選びます。CLIでは起動時の指定として`--device cuda`、`--device mps`、`--device cpu`も使えます。

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

指示・しきい値・候補・実行デバイスは停止中に変更します。更新間隔は0.5 / 1.0 / 2.0 / 5.0 / 10.0秒から選べます。これは次の取得を始める最短間隔であり、推論が遅い場合に画像をためません。CPUで1フレームに約5秒かかる場合、0.5秒を選んでも結果は0.5秒ごとになりません。取得頻度を抑えたい場合は10.0秒を選びます。「推論」は画像の縮小からモデル応答まで、「処理合計」は画面取得開始から表示までの時間です。

## 5. 動作確認

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
.\.venv\Scripts\python.exe -m tests.real_modes_smoke path\to\cats.jpg
.\.venv\Scripts\python.exe -m tests.gui_modes_smoke
```

静止画を個別に試す場合は`tools/check_model.py`を使います。GPU／CPUや環境の状態は`tools/diagnose.py`で確認できます。実モデルの確認には手元の画像を指定してください。画像ファイルはリポジトリに含まれません。

## 参考資料

- [Liquid AI d1-2B モデルカード](https://huggingface.co/LiquidAI/d1-2B)
- [PyTorch インストール手順](https://pytorch.org/get-started/locally/)
- [Pillow ImageGrab](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html)
