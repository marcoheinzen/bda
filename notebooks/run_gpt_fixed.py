# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

def kill_stale_gpt():
    """Kill any orphaned SNAP GPT java processes from previous crashed kernels."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'snap.*gpt|org.esa.snap'],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split()
        if pids and pids[0]:
            print(f"    WARNING: Found {len(pids)} stale GPT process(es): {pids}")
            subprocess.run(['pkill', '-9', '-f', 'snap.*gpt|org.esa.snap'], timeout=10)
            time.sleep(2)
            # also kill any orphaned java processes
            subprocess.run(['pkill', '-9', '-f', 'java.*snap'], capture_output=True, timeout=5)
            time.sleep(1)
            print(f"    Killed stale GPT processes")
    except Exception:
        pass


def _run_gpt_once(cmd, timeout_minutes, step_label):
    """Single GPT execution attempt. Returns (elapsed, None) on success,
    raises on failure."""
    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    output_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                output_lines.append(line)
                if any(kw in line.lower() for kw in ['%', 'done', 'error', 'warning', 'processing', 'writing']):
                    print(f"    GPT: {line[:120]}")
        proc.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        import os, signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise RuntimeError(f"GPT timed out after {timeout_minutes} minutes ({step_label})")
    except Exception:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()

    elapsed = time.time() - start
    if proc.returncode != 0:
        print(f"    GPT FAILED {step_label} (exit code {proc.returncode})")
        for line in output_lines[-30:]:
            print(f"    > {line}")
        raise RuntimeError(f"GPT failed {step_label} with exit code {proc.returncode}")

    return elapsed


def run_gpt(graph_xml_path, timeout_minutes=120, step_label="", max_retries=2):
    """Run SNAP GPT with a processing graph using anti-hang flags.
    Kills orphaned GPT processes before launching. On errno 12 (OOM):
    aggressively kills all Java/SNAP processes, waits for memory, retries.
    """
    cmd = [GPT_PATH, str(graph_xml_path)] + GPT_FLAGS

    for attempt in range(1, max_retries + 1):
        kill_stale_gpt()

        print(f"    Running GPT {step_label}: {Path(graph_xml_path).name}" +
              (f" (attempt {attempt}/{max_retries})" if attempt > 1 else ""))

        try:
            elapsed = _run_gpt_once(cmd, timeout_minutes, step_label)
            print(f"    GPT {step_label} completed in {elapsed/60:.1f} minutes")
            return elapsed

        except OSError as e:
            if e.errno == 12 or 'Cannot allocate memory' in str(e):
                print(f"    ERRNO 12 (OOM) on attempt {attempt}/{max_retries}: {e}")
                print(f"    Killing ALL Java/SNAP processes and waiting for memory...")
                # aggressive kill: all java, all snap, all gpt
                for pattern in ['java', 'snap.*gpt', 'org.esa.snap']:
                    try:
                        subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True, timeout=5)
                    except Exception:
                        pass
                # wait for memory to free
                wait_secs = 30 * attempt  # 30s first retry, 60s second
                print(f"    Waiting {wait_secs}s for memory to free...")
                time.sleep(wait_secs)
                if attempt == max_retries:
                    raise RuntimeError(f"GPT {step_label} failed with errno 12 (OOM) after {max_retries} attempts")
                continue  # retry
            else:
                raise  # non-OOM OSError, don't retry

        except RuntimeError as e:
            # check if GPT output contains memory errors
            err_str = str(e).lower()
            if 'errno 12' in err_str or 'cannot allocate memory' in err_str or 'outofmemoryerror' in err_str:
                print(f"    Memory error in GPT output on attempt {attempt}/{max_retries}")
                for pattern in ['java', 'snap.*gpt', 'org.esa.snap']:
                    try:
                        subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True, timeout=5)
                    except Exception:
                        pass
                wait_secs = 30 * attempt
                print(f"    Waiting {wait_secs}s for memory to free...")
                time.sleep(wait_secs)
                if attempt == max_retries:
                    raise
                continue
            else:
                raise  # non-memory RuntimeError, don't retry
