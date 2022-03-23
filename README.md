## 実行環境について

実行環境はすべて Docker 上で再現できるようにしています。  
基本的には、Detectron2 の Docker 環境をそのまま使用しています。
環境の詳細は以下を参照してください。  
https://github.com/facebookresearch/unbiased-teacher#installation

- Docker Image の build

      ```shell
      cd docker
      docker build -t detectron2 ./
      ```

- コンテナの起動

      データセット等へのパスは適宜変更してください。

      ```shell
      docker run -it --gpus all --shm-size 8gb \
      -v [path to this repository]:/unbiased-teacher \
      -v [path to coco2017]:/unbiased-teacher/datasets/coco \
      detectron2
      ```

      以下が、こちらの実行例です。

      ```shell
      docker run -it --gpus all --shm-size 8gb \
      -v ~/unbiased-teacher:/unbiased-teacher \
      -v ~/datasets/coco2017:/unbiased-teacher/datasets/coco \
      detectron2
      ```

- データセットの配置

      データセットのファイル配置は以下のようにしてください。

      ```
      unbiased_teacher/
      ├── dataseed/
      ├   └── COCO_supervision.txt
      └── datasets/
      └── coco/
            ├── train2017/
            ├── val2017/
            └── annotations/
                  ├── instances_train2017.json
                  └── instances_val2017.json
      ```

## 実行方法

```shell
python train_net.py \
      --num-gpus 8 \
      --config [cofigs/ 以下のファイルへのパス]
```

- 提案手法を COCO の 1% で行い場合は以下のようになります。

      ```shell
      python train_net.py \
            --num-gpus 8 \
            --config configs/coco_supervision_with_iou_pred/sup1_NoPseudoIoUPred_ClsThres075_seed1_run1.yaml
      ```

- 異なる 1% の分割で実験する場合は `DATALOADER.RANDOM_DATA_SEED` を設定します。

      ```shell
      python train_net.py \
            --num-gpus 8 \
            --config configs/coco_supervision_with_iou_pred/sup1_NoPseudoIoUPred_ClsThres075_seed1_run1.yaml \
            DATALOADER.RANDOM_DATA_SEED 2
      ```

      ここで指定している `RANDOM_DATA_SEED` ですが、`dataseed/COCO_supervision.txt` であらかじめどの画像を教師ありとして学習に用いるかを決めたリストを生成しており、それぞれの分割 1 ~ 5 に対応しています。  
      現時点で、0.01%, 0.1%, 0.5%, 1.0%, 2.0%, 5.0%, 10.0% に対応しています。  
      他の割合で実験する場合は、別途リストを作成する必要があります。

