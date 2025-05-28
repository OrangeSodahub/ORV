import argparse
import os
from huggingface_hub import upload_folder, upload_file, hf_hub_download
from rich.console import Console
from rich.panel import Panel
from rich import box, style
from rich.table import Table

CONSOLE = Console(width=120)


def upload():

    if args.folder_path:

        try:
            if token is not None:
                upload_folder(repo_id=args.repo_id, folder_path=args.folder_path, ignore_patterns=ignore_patterns, path_in_repo=args.path_in_repo, token=token)
            else:
                upload_folder(repo_id=args.repo_id, folder_path=args.folder_path, ignore_patterns=ignore_patterns, path_in_repo=args.path_in_repo)
            table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
            table.add_row(f"Model id {args.repo_id}", str(args.folder_path))
            CONSOLE.print(Panel(table, title="[bold][green]:tada: Upload completed DO NOT forget specify the model id in methods! :tada:[/bold]", expand=False))

        except Exception as e:
            CONSOLE.print(f"[bold][yellow]:tada: Upload failed due to {e}.")
            raise e

    if args.file_path:

        try:
            if token is not None:
                upload_file(
                    path_or_fileobj=args.file_path,
                    path_in_repo=os.path.basename(args.file_path),
                    repo_id=args.repo_id,
                    repo_type='model',
                    token=token
                )
            else:
                upload_file(
                    path_or_fileobj=args.file_path,
                    path_in_repo=os.path.basename(args.file_path),
                    repo_id=args.repo_id,
                    repo_type='model',
                )
            table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
            table.add_row(f"Model id {args.repo_id}", str(args.file_path))
            CONSOLE.print(Panel(table, title="[bold][green]:tada: Upload completed! :tada:[/bold]", expand=False))

        except Exception as e:
            CONSOLE.print(f"[bold][yellow]:tada: Upload failed due to {e}.")
            raise e


def download():

    try:
        if token is not None:
            ckpt_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.file_path,
                token=token
            )
        else:
            ckpt_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.file_path,
            )
        table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
        table.add_row(f"Model id {args.repo_id}", str(args.file_path))
        CONSOLE.print(Panel(table, title=f"[bold][green]:tada: Download completed to {ckpt_path}! :tada:[/bold]", expand=False))

        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            import shutil
            shutil.copy(ckpt_path, os.path.join(args.save_path, args.file_path))

    except Exception as e:
        CONSOLE.print(f"[bold][yellow]:tada: Download failed due to {e}.")
        raise e

    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--folder_path", type=str, default=None, required=False)
    parser.add_argument("--file_path", type=str, default=None, required=False)
    parser.add_argument("--save_path", type=str, default=None, required=False)
    parser.add_argument("--token", type=str, default=None, required=False)
    parser.add_argument("--path_in_repo", type=str, default=None, required=False)
    args = parser.parse_args()

    token = args.token or os.getenv("hf_token", None)
    ignore_patterns = ["**/optimizer.bin", "**/random_states*", "**/scaler.pt", "**/scheduler.bin"]

    if not (args.folder_path or args.file_path):
        raise RuntimeError(f'Choose either folder path or file path please!')

    if len(args.repo_id.split('/')) != 2:
        raise RuntimeError(f'Invalid repo_id: {args.repo_id}, please use in [use-id]/[repo-name] format')
    CONSOLE.log(f"Use repo: [bold][yellow] {args.repo_id}")

    if args.upload:
        upload()

    if args.download:
        download()