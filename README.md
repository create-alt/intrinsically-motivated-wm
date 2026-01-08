# 世界モデルにおける探索「深さ」の制御：内発的報酬の動的重み付け
**Controlling the Depth of Exploration in World Models: Dynamic Weighting of Intrinsic Rewards**

本リポジトリは，世界モデルに基づく強化学習手法![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)を基盤として，  
**内発的報酬が探索挙動（特に想像ロールアウトの進み方／深さ）に与える影響**を調査するための
研究用実装をまとめたものである[1]．

本研究では，内発的報酬を単に探索量を増やす信号としてではなく，  
**探索（exploration）と活用（exploitation）の寄与を動的に調整する要素**として捉え直すことを目的とする．

---

## Base Framework

本実装は，以下の DreamerV3 の公開実装を基盤として構築されている．

- DreamerV3 GitHub: https://github.com/danijar/dreamerv3

世界モデルおよび actor–critic 構造の基本設計は原実装に準拠している．  
本リポジトリは DreamerV3 の再実装だけではなく，研究目的に基づく拡張実装である．

---

## Modifications / 工夫点

本研究では DreamerV3 の標準構成に対して，主に以下の変更を加えている．

### 1. 再構成損失の分布モデル変更（MSE → Normal）

世界モデルの再構成損失について，従来の平均二乗誤差（MSE）ではなく，  
**正規分布に基づく尤度（Normal likelihood）**を用いる．

これにより，再構成誤差の **標準偏差**を不確実性の指標として明示的に扱えるようにし，  
世界モデルの予測不確実性を内発的報酬設計に利用可能とする．

### 2. 内発的報酬の導入（agent.py）

`agent.py` において，外発的報酬とは別に **内発的報酬**を追加した．  
内発的報酬は，世界モデルの再構成誤差の不確実性（標準偏差）に基づいて定義され，  
学習の進行に応じて **探索（exploration）と活用（exploitation）の寄与が変化**するよう設計されている．

---

## Experimental Setting

実験は **Atari-100K 設定**下の一部の Atari 環境を対象として行っている．  
現時点では予備的な実験段階であり，複数環境に対する網羅的な評価や最終性能の比較を目的としていない．

---

## Disclaimer

本リポジトリは学術研究および教育目的のためのものである．  
DreamerV3 およびその原実装に関する権利はすべて原著者に帰属する．  
本研究は，内発的報酬と探索挙動の関係を分析するための実験的・探索的な実装を提供する．

---

## Reference

[1] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023).  
*Mastering diverse domains through world models*, 2024.  
URL: https://arxiv.org/abs/2301.04104
