# Chạy lab trên Kaggle

Nguyễn Văn Thân — GitHub `meth04`, Kaggle `soobinms`.

Kaggle CLI đã cài cho Python của người dùng. Dùng `python -m kaggle` kể cả khi terminal hiện tại chưa cập nhật PATH. Lệnh `kaggle` có sẵn trong terminal mở mới. Đăng nhập OAuth lưu ngoài repo, không cần đặt khóa trong notebook hoặc `.env`.

Huấn luyện, suy luận, dữ liệu và trọng số nằm trên Kaggle. Máy cá nhân chỉ sửa mã, đóng gói source, gửi lệnh và nhận báo cáo nhỏ. Một T4 được dùng dù Kaggle cấp T4 x2; batch=2, grad_accum=4 đúng config của đề. Loss chỉ tạo logits cho token được giám sát, giữ nguyên causal cross entropy và kiểm tra tương đương trên GPU. Box dưới độ phân giải tọa độ được mở tối thiểu một đơn vị, giữ nhãn và GT gốc. Nền tảng Kaggle khác yêu cầu Colab của đề; báo cáo ghi rõ thay đổi này.

## Chạy tự động

```powershell
python tools/run_campaign.py --smoke-already-submitted --cloud-workflow
```

Bộ điều khiển đợi smoke test và E0 hoàn tất, rồi gửi notebook `full` lên Kaggle. Notebook này tự chạy E1, E2, E3, E4, E5 và tạo báo cáo trong cùng một phiên; dùng lại model gốc từ output E0. Nếu E1 đã được gửi riêng, bộ điều khiển chờ E1 và gắn output đó vào `full` để tái sử dụng, không chạy test E1 lần nữa. Sau khi `full` đã được gửi, việc huấn luyện không cần máy cá nhân gửi lệnh giữa các phase. Khi máy cá nhân kết nối lại, CLI có thể tiếp tục lấy kết quả, kiểm tra rubric, commit và đẩy lên GitHub. Trạng thái lưu ở `work/campaign/state.json`; có thể chạy lại lệnh để tiếp tục theo dõi. Phase lỗi dừng chiến dịch để sửa; không tự sinh hoặc thay thế số liệu.

Mỗi phiên Kaggle có giới hạn thời gian; nếu workflow không kịp hoàn tất, adapter/checkpoint và các phase đã xong vẫn nằm trong output riêng tư để tiếp tục. Các setup dùng cùng một T4 và batch hiệu dụng tám ảnh. Bỏ `--cloud-workflow` nếu cần chia các phase thành notebook riêng, tối đa hai notebook đồng thời; cách này cần CLI trên máy cá nhân gửi từng phase tiếp theo.

E4 chọn `data_size`: 10% và 100%. Baseline tốt nhất được chọn bằng validation. Điều kiện 100% tái sử dụng đúng baseline; điều kiện 10% huấn luyện riêng 300 bước. E5 dùng TRL 1.9.2, GRPO 60 bước, reward IoU đã có test.

## Điều khiển từng lượt

```powershell
python tools/kaggle_control.py status --phase smoke
python tools/kaggle_control.py log --phase smoke
python tools/kaggle_control.py push --phase e0
python tools/kaggle_control.py push --phase e1
python tools/kaggle_control.py push --phase e2
python tools/kaggle_control.py push --phase e3
python tools/kaggle_control.py push --phase e4 --source soobinms/day10-qwen3vl-e1 --source soobinms/day10-qwen3vl-e2 --source soobinms/day10-qwen3vl-e3
python tools/kaggle_control.py push --phase e5 --source soobinms/day10-qwen3vl-e1 --source soobinms/day10-qwen3vl-e2 --source soobinms/day10-qwen3vl-e3
```

Các notebook đều riêng tư. GPU/account quota và lỗi runtime có thể làm chiến dịch dừng. Lệnh `log` đọc log của lượt đã có output; khi kernel còn chạy có thể chưa có log.

## Mã và kiểm tra

- `solution/data.py`: JSON, thang tọa độ, chat template, response-only loss và collator.
- `solution/modeling.py`: NF4, phạm vi LoRA, projector float32, checkpoint riêng và hash khi nạp lại.
- `solution/engine.py`: huấn luyện, chọn checkpoint bằng validation, suy luận và chấm bằng công cụ gốc.
- `solution/reward.py`: reward IoU; `solution/test_solution.py` kiểm tra hình học và reward, không cần GPU.
- `solution/run.py`: từng phase trên Kaggle, từ chối huấn luyện ở máy cá nhân.
- `solution/reporting.py`: báo cáo, sáu ảnh minh họa, so sánh CI và kiểm tra reward hacking từ kết quả thật.

```powershell
python -m pytest -q tests solution/test_solution.py
python scripts/check_submission.py
python tools/check_rubric.py
```

Trước khi có các lượt chạy thật, hai lệnh kiểm tra bài nộp sẽ báo thiếu bằng chứng/báo cáo. Sau khi điền báo cáo, test gốc `test_report_template_has_student_markers_and_ten_questions` vẫn yêu cầu `<điền` vì nó kiểm tra *mẫu đề*, trái với yêu cầu bài nộp hoàn chỉnh. Không sửa test gốc; kiểm tra các test còn lại với `-k "not report_template_has_student_markers_and_ten_questions"`, rồi dùng hai bộ kiểm tra bài nộp để xác nhận báo cáo hoàn chỉnh.

Không commit `data/`, `work/`, credential hay trọng số. Bộ điều khiển chỉ tải báo cáo, evidence, log và ảnh minh họa; dữ liệu ảnh và adapter được giữ trên Kaggle.
