# Visual Detector

選択した画面領域をローカルのLiquid AI d1-2Bで解析するデスクトップアプリです。**領域検出・全体判定・全体推論**の3モードを備えます。「全体」は、選択した画面領域の画像全体を指します。

## 起動

モデル重みと仮想環境はリポジトリに含まれません。PythonとTkinterを用意し、PyTorchを[公式手順](https://pytorch.org/get-started/locally/)に従って導入してください。Windowsでの準備例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe tools\download_model.py --accept-model-code
.\run_local.bat
```

GPUを使う場合は、その環境に合ったCUDA版PyTorchの導入コマンドを先に確認してください。モデル取得前に[Liquid AIのモデルライセンスと独自コード](https://huggingface.co/LiquidAI/d1-2B)を確認してください。macOS/Linuxの手順と詳しい操作は[構築・操作手順書](docs/BUILD_GUIDE_JA.html)にあります。

1. 小窓右上の「実行」でGPUまたはCPUを選びます。
2. 「領域選択」で主画面の監視範囲をドラッグし、小窓をその範囲の外へ移動します。
3. モードと指示を設定して「開始」を押します。停止中にGPU／CPUを変更すると、次の開始時にモデルを読み直します。

GPUが利用できない環境ではCPUを選択します。CPUとApple Silicon MPSはfloat32、CUDAは対応状況に応じてbfloat16またはfloat16を使います。コマンドからは`--device cuda`、`--device mps`、`--device cpu`も指定できます。

## モード

| モード | 動作 |
|---|---|
| 領域検出 | 指示に合う部分をBBoxで囲みます。初期指示は「映っている魚をすべて囲む」。信頼度下限は0.00〜1.00です。 |
| 全体判定 | 選択領域全体を二値判定します。初期指示は「魚が映っているか？」、判定しきい値は0.20。スコアがしきい値を超えた陽性フレームを、新しい順に3枚まで取得時刻付きで表示します。「位置の指示」の初期値は「映っている魚をすべて囲む」で、陽性の後だけ別の推論でBBoxを探します。空欄なら位置は探しません。 |
| 全体推論 | 「候補を編集」で分類候補を1行ずつ入力します。初期候補は「魚がある」のみ。「該当しない」は自動追加され、最高スコアの1件を各フレームに表示します。 |

しきい値、指示、候補、モードは停止中に変更できます。全体判定の陽性履歴は停止後も残り、再開・領域変更・モード変更で消去されます。

## 更新間隔と処理時間

更新間隔は**0.5 / 1.0 / 2.0 / 5.0 / 10.0秒**から選べます。これは画面取得を開始する間隔の下限です。処理が長引いた場合は完了後に最新の画面を取得し、途中のフレームをためません。CPUで1フレームに約5秒かかる場合、0.5秒を選んでも0.5秒ごとの結果にはなりません。取得頻度を抑えたい場合は10.0秒を選んでください。

画面の「推論」は画像縮小からモデル応答まで、「処理合計」は画面取得開始から表示までの時間です。小窓が選択領域に重なると取得を一時停止します。

## コード構成

| ファイル | 役割 |
|---|---|
| `desktop_detector/core.py` | 推論設定、応答の検証、座標計算、取得間隔の制御 |
| `desktop_detector/backend.py` | ローカルモデルの読込と3モードの推論 |
| `desktop_detector/worker.py` | 推論専用スレッドとフレーム処理 |
| `desktop_detector/capture.py` | 主画面の画像取得 |
| `desktop_detector/selection.py` | 監視領域の選択画面 |
| `desktop_detector/preview.py` | 結果画像とBBox・陽性履歴の描画 |
| `desktop_detector/ui.py` | 小窓の操作と各処理の連携 |

推論には`model.system_one()`の`boxes`・`noul`・`choice`を使用します。HTTP推論サーバーやデータベースは使いません。

## 確認

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -q
.\.venv\Scripts\python.exe -m tests.gui_modes_smoke
.\.venv\Scripts\python.exe -m tests.real_modes_smoke path\to\cats.jpg
```

単体テストとGUI統合テストのほか、手元の猫画像を渡して実モデルの応答を確認できます。サンプル画像は同梱していません。Windows環境でCUDA・CPU推論とGPUからCPUへの切り替えを確認済みです。魚の検出精度と長時間運転は未測定です。

モデルの仕様：[Liquid AI d1-2B](https://huggingface.co/LiquidAI/d1-2B)。
