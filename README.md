# Vietnamese-Tone-Prediction-with-Transformer-Decoder
Solution for AIviVN's Vietnamese tone prediction competition

# Overview

Đây là một baseline cho bài toán thêm dấu tiếng Việt của AIviVN sử dụng mô hình Transformer.

Phần lớn code cho mô hình được lấy từ repo này của Kyubyong Park: https://github.com/Kyubyong/transformer .

Ở đây có một số thay đổi, cụ thể:

- Mình bỏ phần encoder đi, chỉ giữ lại 1 nửa decoder. Do bài toán này là map 1-1 từ input về output nên mô hình sẽ không cần autoregressive deocoding nữa.
- Không sử dụng label smoothing cho hàm loss vì sẽ làm model hội tụ chậm hơn.

# Trainng 

Để chuẩn bị training bạn cần chuẩn bị các file sau:

## Tập training

Cần 2 file ```train.src``` và ```train.tgt```. Trong đó file ```train.tgt``` chính là file train mà ban tổ chức đã cho, và file ```train.src``` là sau khi đã bỏ dấu, bạn sẽ dùng hàm ```remove_tone_file()``` trong file ```utils.py``` để sinh file này.

## Tập test

Do file test của btc bị lỗi nên mình đã tự clean lại bằng cách bỏ đi 1 số token tự nhiên mất đi và lưu lại thành file ```test_cleaned.txt```.

## Tập từ điển (vocabulary)

Mình chỉ giữ lại các âm tiết thuần việt trong file training và phiên bản không dấu tương ứng của chúng lưu lại thành 2 file ```tgt.vocab.tsv``` và ```src.vocab.tsv``` .
Bạn có thể tạo file vocab riêng của mình theo format đó.

## Quá trình training/test

Các hyperparameter được lưu lại trong file ```hyperparams.py```. Để training bạn chỉ cần chạy:

```python train.py```

Để sinh file kết quả chạy:

```python eval.py```

Kết quả sẽ sinh ra trong file ```results/logdir.txt``` . Bạn dùng hàm ```decompose_predicted_test_file()``` trong file ```utils.py``` để gen ra kết quả.


