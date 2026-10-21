"""Read-only live desktop progress window for the corrected PDF benchmark."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'benchmark/revised_pdf_corrected_20261021'
METHODS = {'cdirp_v3': 'CDIRP V3', 'gpt6_luna': 'GPT-6 Luna · low',
           'claude_haiku55': 'Claude Haiku 5.5 · low', 'gemini38_flash': 'Gemini 3.8 Flash · low'}


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}


def snapshot(root):
    manifest = read(root / 'manifest.json')
    batches = manifest.get('baseline_retrieval', {}).get('batches_per_question', 18)
    result = []
    for method, label in METHODS.items():
        cells, errors, success, fallback, failed, selected = [], [], 0, 0, 0, 0
        for i in range(1, 10):
            folder = root / 'runs' / method / f'Q{i}'
            status = read(folder / 'status.json')
            state = status.get('status')
            count = len(list((folder / 'calls').glob('retrieve_*/accepted.json')))
            selected += count
            if state == 'completed':
                success += 1
                cell = '完成'
            elif state == 'failed':
                is_fallback = str(status.get('error', '')).startswith('Production returned source-list fallback:')
                fallback += int(is_fallback)
                failed += int(not is_fallback)
                cell = '降级输出' if is_fallback else '失败'
                errors.append(f"{label} Q{i}: {status.get('error', 'Unknown error')}")
            elif state == 'running':
                cell = ('检索/生成中' if method == 'cdirp_v3' else
                        '生成答案中' if count == batches else f'扫描 {count}/{batches}')
            else:
                cell = f'已存 {count}/{batches}' if count else '待运行'
            cells.append(cell)
        failure = read(root / 'runs' / method / 'failure.json')
        if failure:
            errors.append(f"{label} 运行记录: {failure.get('error', failure)}")
        result.append({'method': method, 'label': label, 'success': success,
                       'fallback': fallback, 'failed': failed, 'batches': selected,
                       'cells': cells, 'errors': errors})
    return manifest, batches, result


def show(root):
    window = tk.Tk()
    window.title('PDF 实验实时进度 — 四模型重跑')
    window.geometry('1280x650')
    window.minsize(1050, 550)
    style = ttk.Style(window)
    style.configure('Treeview', rowheight=42, font=('Microsoft YaHei UI', 10))
    style.configure('Treeview.Heading', font=('Microsoft YaHei UI', 10, 'bold'))
    outer = ttk.Frame(window, padding=18)
    outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='修正 PDF 后的四模型实验', font=('Microsoft YaHei UI', 18, 'bold')).pack(anchor='w')
    ttk.Label(outer, text=str(root), foreground='#555555').pack(anchor='w', pady=(4, 12))
    headline = ttk.Label(outer, font=('Microsoft YaHei UI', 11))
    headline.pack(anchor='w', pady=(0, 10))
    columns = ['model', 'done', 'scan'] + [f'Q{i}' for i in range(1, 10)]
    table = ttk.Treeview(outer, columns=columns, show='headings', height=4)
    for column in columns:
        title = {'model': '模型', 'done': '成功 / 降级 / 失败', 'scan': '检索批次'}.get(column, column)
        table.heading(column, text=title)
        table.column(column, width=170 if column == 'model' else 135 if column == 'done' else 85, anchor='center')
    table.pack(fill='x')
    for method in METHODS:
        table.insert('', 'end', iid=method)
    ttk.Label(outer, text='每 5 秒刷新；扫描批次完成后才生成答案。降级 ≠ 成功。V3 使用原生检索，不按 18 批扫描。',
              wraplength=1150).pack(anchor='w', pady=(10, 4))
    ttk.Label(outer, text='状态来自磁盘检查点；“运行中”表示尚未写入结束状态，不是进程存活保证。关闭本窗口不会停止实验。',
              wraplength=1150).pack(anchor='w', pady=(0, 8))
    handoff = ttk.Label(outer, font=('Microsoft YaHei UI', 11, 'bold'))
    handoff.pack(anchor='w', pady=6)
    ttk.Label(outer, text='已记录的题目失败 / 降级原因：').pack(anchor='w')
    error_frame = ttk.Frame(outer)
    error_frame.pack(fill='both', expand=True)
    errors = tk.Text(error_frame, wrap='word', height=8, font=('Consolas', 10))
    scroll = ttk.Scrollbar(error_frame, command=errors.yview)
    errors.configure(yscrollcommand=scroll.set)
    errors.pack(side='left', fill='both', expand=True)
    scroll.pack(side='right', fill='y')
    previous_errors = None

    def refresh():
        nonlocal previous_errors
        manifest, batches, rows = snapshot(root)
        headline.configure(text=f"更新时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}    "
                                f"语料：{manifest.get('corpus_documents', '?')} PDFs / {manifest.get('corpus_pages', '?')} 页    "
                                f"实验状态：{manifest.get('status', '未读取到')}")
        for row in rows:
            scan = '原生检索' if row['method'] == 'cdirp_v3' else f"{row['batches']}/{9*batches}"
            table.item(row['method'], values=[row['label'], f"{row['success']} / {row['fallback']} / {row['failed']}", scan, *row['cells']])
        export = read(root / 'handoff_export_status.json')
        labels = {'waiting_for_inference': '等待模型运行结束', 'validating_and_exporting': '正在校验并打包',
                  'completed': '已完成校验和打包', 'incomplete_or_export_error': '未完成或导出失败，请检查日志', 'wait_timeout': '等待超时'}
        state = export.get('status', '未开始')
        handoff.configure(text='交付包：' + labels.get(state, state))
        text = '\n\n'.join(e for row in rows for e in row['errors']) or '暂无题目级失败记录。'
        if text != previous_errors:
            errors.configure(state='normal')
            errors.delete('1.0', 'end')
            errors.insert('1.0', text)
            errors.configure(state='disabled')
            previous_errors = text
        window.after(5000, refresh)

    refresh()
    window.lift()
    window.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.once:
        _, batches, rows = snapshot(args.root.resolve())
        print(json.dumps({'batches_per_question': batches, 'models': rows}, ensure_ascii=True, indent=2))
    else:
        show(args.root.resolve())
