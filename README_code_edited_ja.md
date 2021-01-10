[オリジナル](https://github.com/qqwweee/keras-yolo3)から以下の変更を行っています。

# 1. `yolo_video.py`の修正

YOLOv3のKeras版実装では、`yolo_video.py`の引数名と`yolo.py`の引数名が不一致であるため、`yolo_video.py`で受け取った引数を`yolo.py`で受け取ることができていません。

`yolo_video.py`を次のように変更し、引数（model_path, anchors_path, classes_path）を受け取れることができるようにします。これに従い引数名も変更となります。

**`yolo_video.py`の一部（変更前）**

```python
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
```

**`yolo_video.py`の一部（変更後）**

```python
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
```

参考：[ArgumentParserの使い方を簡単にまとめた](https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0)

# 2. `voc_annotation.py`の修正

> VoTTで出力したファイルには、アノテーション位置がfloatで書かれています。そのままでは`voc_annotation.py`で読み込めないので、下記のようにfloatからintに変換するようにソースを修正します。

引用：https://qiita.com/moto2g/items/dde7a55fceda862b2390

**`voc_annotation.py`の一部（変更前）**

```python
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
```

**`voc_annotation.py`の一部（変更後）**

```python
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))
```

# 3. `train.py`の変更

`train.py`で「tensorflow.python.framework.errors_impl.ResourceExhaustedError: 2 root error(s) found. 」というエラーが表示され最後まで処理が行われない場合は、[GPUのメモリが足りない可能性](https://qiita.com/enoughspacefor/items/1c09a27877877c56f25a)があります。

以下の個所の`batch_size`を、エラーがでなくなるまで小さくする必要があります。
（batch_size指定は2か所ありますが、76行目の「note that more GPU memory is required after unfreezing the body」というコメントが記載されている個所となります。）

**`train.py`の一部（変更前）**

```python
        batch_size = 32  # note that more GPU memory is required after unfreezing the body
```

以下では`8`に変更していますが、GPUのメモリエラーが出なくなるまで小さくします。
私の環境（NVIDIA GeForce GTX 1080 Ti）では`32`→`8`まで減らす必要がありました。

**`train.py`の一部（変更後）**

```python
        batch_size = 8  # note that more GPU memory is required after unfreezing the body
```

# 4. プルリクエスト[Training (add tensorboard debug, and mAP Calculation) #206](https://github.com/qqwweee/keras-yolo3/pull/206)の反映

TensorBoardへの対応、[mAP](https://qiita.com/tmtakashi_dist/items/863e1781b5252e453b47)への対応を行った[プルリクエスト](https://github.com/qqwweee/keras-yolo3/pull/206)の反映と関連する設定について記載します。
（[YOLOv3のKeras版実装](https://github.com/qqwweee/keras-yolo3)の最終更新は2年ほど前のためか、上記プルリクエストは反映されていません）

1. `tensorboard_logging.py`をダウンロードして追加します。

    [tensorboard_logging.py](https://github.com/qqwweee/keras-yolo3/blob/f4a9c40f4615cdbb774942507ecad3af5f05c990/tensorboard_logging.py)（[Raw]ボタンの結果を保存する）

    ```
    keras-yolo3\tensorboard_logging.py
    ```

2. `train_v2.py`をダウンロードして追加します。

    [train_v2.py](https://github.com/qqwweee/keras-yolo3/blob/f4a9c40f4615cdbb774942507ecad3af5f05c990/train_v2.py)（[Raw]ボタンの結果を保存する）

    ```
    keras-yolo3\train_v2.py
    ```

3. `model.py`をダウンロードして**上書き**します。

    [model.py](https://github.com/qqwweee/keras-yolo3/blob/f4a9c40f4615cdbb774942507ecad3af5f05c990/yolo3/model.py)（[Raw]ボタンの結果を保存する）

    ```
    keras-yolo3\yolo3\train_v2.py
    ```

# 5. リサイズ用スクリプトの追加

リサイズ用スクリプトとして、ここではsleeplessさんの[keras-yolo3のフォーク](https://github.com/sleepless-se/keras-yolo3)の`resize_images.py`を参考に、**一部変更して**使用します。

`resize_images.py`をダウンロードし以下に配置します。

```
keras-yolo3\resize_images.py
```

[元のソース](https://github.com/sleepless-se/keras-yolo3/blob/master/resize_images.py)から以下を変更しています。

- リサイズ後の大きさ`image_size = 320`を、`yolov3.cfg`に合わせる形で`image_size = 416`に変更
- リサイズに使用する関数を、推論時と同じ`rgb_im.resize`に変更
- 正方形ではない画像を正方形に収まるようにリサイズする位置を、推測時と合わせる形で中央に変更
- 正方形になるように埋める背景色を、推論時と合わせる形で灰色（128,128,128）に変更

**resize_images.py（image_sizeを416に変更済み）**

```python
import os
import sys
from glob import glob
from PIL import Image

def resize_images(images_dir, image_save_dir, image_size):
    os.makedirs(image_save_dir, exist_ok=True)

    img_paths = glob(os.path.join(images_dir, '*'))

    for img_path in img_paths:
        # image open
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')

        # resize image with unchanged aspect ratio using padding
        iw, ih = image.size
        w, h = (image_size,image_size)
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        # resize
        rgb_im = rgb_im.resize((nw,nh), Image.BICUBIC)

        # make background
        back_ground = Image.new("RGB", (image_size,image_size), color=(128,128,128))
        back_ground.paste(rgb_im, ((w-nw)//2, (h-nh)//2))

        # make path
        save_path = os.path.join(image_save_dir, os.path.basename(img_path))
        end_index = save_path.rfind('.')
        save_path = save_path[0:end_index]+'.png'
        print('save',save_path)
        back_ground.save(save_path,format='PNG')

def _main():
    images_dir = 'images/'  # input directory
    image_save_dir = 'resize_image/'  # output directory
    image_size = 416
    if len(sys.argv) > 1:
        image_size = int(sys.argv[1])

    resize_images(images_dir=images_dir, image_save_dir=image_save_dir, image_size=image_size)

if __name__ == '__main__':
    _main()
```
参考：https://github.com/sleepless-se/keras-yolo3/blob/master/resize_images.py


# 6. `voc_annotation.py`の変更

「5. リサイズ用スクリプトの準備」で出力形式をPNGに変更したことに伴い、`voc_annotation.py`を変更します。

**`voc_annotation.py`の一部（変更前）**

```python:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
```

**`voc_annotation.py`の一部（変更後）**

```python:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
```

# 7. `train.py`の変更（voc_annotation.pyとtrain.py間のファイル名不一致に関する対処）

以下のように、`voc_annotation.py`で出力されるファイル名と`train.py`で定義されているファイルが不一致となっています。

|ソース|ファイル名|
|---|---|
|`voc_annotation.py`|`2007_train.txt`|
|`train.py`|`train.txt`|

以下では、`train.py`を変更していますが、`voc_annotation.py`の変更を検討してもよろしいかと思います。

`voc_annotation.py`で出力されるファイル名に合わせて、`train.py`のソースを変更します。

**`train.py`の一部（変更前）**

```python:
annotation_path = 'train.txt'
```

**`train.py`の一部（変更後）**

```python:
annotation_path = '2007_train.txt'
```

# 8. YOLOv3-tinyに関する変更

[YOLOv3のKeras版実装](https://github.com/qqwweee/keras-yolo3)では、YOLOv3-tiny版のアンカーファイルの扱い方について、議論があるようです。Pull Request([503](https://github.com/qqwweee/keras-yolo3/pull/503),[622](https://github.com/qqwweee/keras-yolo3/pull/622))、Issue([306](https://github.com/qqwweee/keras-yolo3/issues/306),[428](https://github.com/qqwweee/keras-yolo3/issues/428),[512](https://github.com/qqwweee/keras-yolo3/issues/512),[599](https://github.com/qqwweee/keras-yolo3/issues/599),[625](https://github.com/qqwweee/keras-yolo3/issues/625))が上げられています。しかしYOLOv3のKeras版実装の最終更新は2年ほど前のためか、リポジトリへの反映は行われていません。

[Darknet側の修正](https://github.com/pjreddie/darknet/commit/f86901f6177dfc6116360a13cc06ab680e0c86b0#diff-2b0e16f442a744897f1606ff1a0f99d3)も踏まえ、最小のアンカーボックスを使う場合の反映方法を示します。

**`model.py`の一部（変更前）**
```python
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
```

**`model.py`の一部（変更後）**

```python
 anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
 ```

# 9. 動画変換時の問題への対処

## [YOLOv3で動画入力時最終フレームの処理後AttributeErrorが出る時の対処法](https://qiita.com/toten_s/items/04d0744598e8c3d9bc9b)の問題に対処

OpenCV 3系（Python 3.6）の環境では、最終フレームの認識が後にエラーが表示されることがあり、`AttributeError: 'NoneType' object has no attribute '__array_interface__'`というエラーが表示されます。


**`yolo.py`の一部（変更前）**
```python
     while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
```

**`yolo.py`の一部（変更後）**
```python
     while True:
        return_value, frame = vid.read()
        if type(frame) == type(None): break
        image = Image.fromarray(frame)
```

## 動画が出力されない

OpenCV 3系（Python 3.6）の環境では、MP4Vをavc1と認識することがあるため、`int(cap.get(cv2.CAP_PROP_FOURCC))`による動画形式の自動認識がうまくいきません。

**`yolo.py`の一部（変更前）**
```python
video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
```

**`yolo.py`の一部（変更後）**
```python
# video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
# print(int(vid.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('utf-8'))
video_FourCC    =  cv2.VideoWriter_fourcc(*'MP4V') #OpenCV 3(Python 3.6)では、MP4Vがavc1と認識されるので固定値で設定
```

# 10. TensorBoardへの対応したtrain_v2.pyをyolov3-tiny 対応に変更

[ TensorBoard対応のプルリクエスト](https://github.com/qqwweee/keras-yolo3/pull/206)はyolov3-tiny 対応ではなかったため、対応するように変更（Y.A.さんに感謝します。）

https://github.com/tfukumori/keras-yolo3/commit/564cce259df453366b64921f34412e5dcd65cc64

具体的には以下の変更を行っている（[@caramel_3]https://qiita.com/caramel_3) さんに感謝します）。

- anchors_file引数のデフォルト値を削除（`model_data/yolo_anchors.txt`、または`model_data/yolo_anchors.txt` 指定する）
- アンカーファイルのアンカー数に応じて使用するモデルを変更
- tinyモデルの読み込み関数を追加
- 重みに加えてモデルも保存するように変更
- アウトプットをtinyモデルに対応（出力数がも異なる）
- TensorBoard用のlogファイルをtinyモデルに対応

# 11. gitignoreへの追加

gitでソース管理する場合は、以下を`.gitignore`に追加します。

```:.gitignoreに追加
# Custom
images/
resize_image/
VOCdevkit/
2007_test.txt
2007_train.txt
2007_val.txt
yolo_logs/
tmp_gt_files/
tmp_gt_files_org/
tmp_pred_files/
.vscode/
```

# 12. camera対応

https://github.com/tfukumori/keras-yolo3/blob/master/yolo_webcam.py

接続したカメラに対してリアルタイムで認識を行う際に使用します。
オプションを指定することでファイル保存を行うこともできます。

接続したカメラへの認識のソースは以下を参考にしています。
https://qiita.com/yoyoyo_/items/10d550b03b4b9c175d9c

また、リアルタイムの認識ということでFPSが一定ではないため結果をCBRとして保存するための方法としては、以下を参考にしています。
https://madeinpc.blog.fc2.com/?no=1364
