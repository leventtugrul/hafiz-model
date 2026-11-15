#!/usr/bin/env python3
"""Fonem modeli laboratuvarı (CI): IqraEval wav2vec2 CTC modelini ONNX'e çevir,
gerçek ve bozulmuş tilavet örneklerinde tahmin edilen fonemleri beklenenle karşılaştır."""
import os, sys, json, glob, argparse, subprocess
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from uthmani import uth2msa

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt"); ap.add_argument("--dict"); ap.add_argument("--phonetiser")
ap.add_argument("--data"); ap.add_argument("--audio"); ap.add_argument("--out")
a = ap.parse_args()
sys.path.insert(0, a.phonetiser)
from phonetiser.phonetise_Arabic import phonetise

def phon(text):
    r = phonetise(uth2msa(text))
    return [p.rstrip("01") for p in " ".join(r[1]).split() if p != "sil"]

# ---- s3prl modelini kur (resmi çıkarım betiğiyle aynı yol) ----
from s3prl.downstream.runner import Runner
import argparse as _ap
torch.serialization.add_safe_globals([_ap.Namespace])
md = torch.load(a.ckpt, map_location="cpu", weights_only=False)
args, cfg = md["Args"], md["Config"]
args.init_ckpt = a.ckpt; args.device = "cpu"
cfg["downstream_expert"]["text"]["vocab_file"] = a.dict
cfg["runner"]["upstream_finetune"] = False; cfg["runner"]["layer_drop"] = False
cfg["runner"]["downstream_pretrained"] = None
runner = Runner(args, cfg)
up, fe, down = runner._get_upstream(), runner._get_featurizer(), runner._get_downstream()
for m in (up.model, fe.model, down.model): m.eval()
exp = down.model
print("downstream alanları:", [k for k in vars(exp) if not k.startswith("_")][:40])
print("modüller:", [n for n, _ in exp.named_children()])
print(exp)
tok = exp.tokenizer
vocab = [tok.decode([i]) if hasattr(tok, "decode") else str(i) for i in range(tok.vocab_size)]
print("vocab_size", tok.vocab_size, "pad", getattr(tok, "pad_idx", None), "ilk 10:", vocab[:10])

# ONNX dışa aktarımı için: fairseq pad_to_multiple izlemede tensörle çalışmıyor
import s3prl.upstream.wav2vec2.wav2vec2_model as w2m
_orig_pad = w2m.pad_to_multiple
def _pad(x, multiple, dim=-1, value=0):
    if x is None or multiple <= 1 or torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
        return x, 0
    return _orig_pad(x, multiple, dim, value)
w2m.pad_to_multiple = _pad

# tek örnek için izlenebilir pad_sequence (s3prl modüllerinde yerine koy)
import torch.nn.utils.rnn as _rnn
_orig_ps = _rnn.pad_sequence
def _ps(seqs, batch_first=False, padding_value=0.0, **kw):
    if len(seqs) == 1:
        x = seqs[0].unsqueeze(0)
        return x if batch_first else x.transpose(0, 1)
    return _orig_ps(seqs, batch_first=batch_first, padding_value=padding_value, **kw)
for _name, _mod in list(sys.modules.items()):
    if _name.startswith("s3prl") and getattr(_mod, "pad_sequence", None) is _orig_ps:
        _mod.pad_sequence = _ps; print("pad_sequence değiştirildi:", _name)

class Head(torch.nn.Module):
    """s3prl RNNs başlığının paketlemesiz eşdeğeri (tek örnek)."""
    def __init__(s, ex):
        super().__init__(); s.proj = ex.projector; s.rnns = ex.model.rnns; s.lin = ex.model.linear
    def forward(s, x):
        x = s.proj(x)
        for l in s.rnns:
            x, _ = l.layer(x)
            x = torch.tanh(l.pj(x))
        return s.lin(x)

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.up, s.fe, s.ex = up.model, fe.model, exp
    def forward(s, wav):                      # wav: [1, N]
        w = [wav[0]]
        f = s.fe(w, s.up(w))                  # list of [T, D]
        x = f[0].unsqueeze(0)
        x = s.ex.projector(x)
        lens = torch.tensor([x.shape[1]], dtype=torch.long)
        logits, _ = s.ex.model(x, lens)
        return logits                          # [1, T, C]

class NetX(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.up, s.fe, s.h = up.model, fe.model, Head(exp)
    def forward(s, wav):
        w = [wav[0]]
        f = s.fe(w, s.up(w))
        return s.h(f[0].unsqueeze(0))

# s3prl sarmalayıcısı uzunlukları Python sayısı olarak sabitliyor; doğrudan iç modeli kullan
inner = up.model.model
print("iç model:", type(inner).__name__, "katman:", len(inner.encoder.layers), "layer_norm_first:", getattr(inner.encoder, "layer_norm_first", None))
print("featurizer:", {k: v for k, v in vars(fe.model).items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))})

class NetY(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.m = inner; s.h = Head(exp)
        s.w = torch.nn.Parameter(fe.model.weights.detach().clone(), requires_grad=False)
        s.norm = bool(getattr(fe.model, "normalize", False))
        s._buf = []
        s.m.encoder.register_forward_hook(lambda mod, inp, out: None)
        s.m.encoder.register_forward_pre_hook(lambda mod, inp: s._buf.append(inp[0]))
        for L in s.m.encoder.layers:
            L.register_forward_hook(lambda mod, inp, out: s._buf.append(out[0].transpose(0, 1)))
    def forward(s, wav):
        s._buf.clear()
        s.m.extract_features(wav, padding_mask=None, mask=False)
        hs = s._buf[:13]
        T = hs[0].shape[1]
        hs = [h[:, :T] for h in hs]                  # kodlayıcının iç dolgusunu kırp
        st = torch.stack(hs, 0)                      # [13, 1, T, D]
        if s.norm: st = torch.nn.functional.layer_norm(st, (st.shape[-1],))
        w = torch.softmax(s.w, -1).view(-1, 1, 1, 1)
        x = (w * st).sum(0)                          # [1, T, D]
        return s.h(x)

net = Net().eval()
netx = NetY().eval()
with torch.no_grad():
    for sec in (3, 7):
        _w = torch.randn(1, 16000 * sec) * 0.05
        a1, a2 = net(_w), netx(_w)
        print(f"eşdeğerlik ({sec} sn): şekil {tuple(a1.shape)} vs {tuple(a2.shape)}, fark {float((a1 - a2).abs().max()):.2e}")

def greedy(lg):
    ids = lg.argmax(-1).tolist(); out, prev = [], -1
    blank = getattr(tok, "pad_idx", 0)
    for i in ids:
        if i != prev and i != blank: out.append(vocab[i])
        prev = i
    return out

cases = json.load(open(os.path.join(a.audio, "cases.json"))) if a.audio and os.path.exists(os.path.join(a.audio or "", "cases.json")) else []
data = json.load(open(os.path.join(a.data, "s", "001.json"))) if cases else None
os.makedirs(a.out, exist_ok=True)
with torch.no_grad():
    for c in cases:
        x = np.fromfile(os.path.join(a.audio, c["file"]), dtype=np.float32)
        lg = net(torch.from_numpy(x)[None])
        pred = greedy(lg[0])
        exp_ = phon(" ".join(data["ayahs"][i - 1]["t"] for i in c["ayahs"]))
        print(f"== {c['name']}\n   beklenen: {' '.join(exp_)}\n   duyulan : {' '.join(pred)}")

# ---- ONNX ----
wav = torch.zeros(1, 16000 * 4)
try:
    torch.onnx.export(netx, (wav,), os.path.join(a.out, "phoneme.onnx"), input_names=["wav"], output_names=["logits"],
                      dynamic_axes={"wav": {1: "n"}, "logits": {1: "t"}}, opset_version=17, do_constant_folding=True)
    print("ONNX tamam", os.path.getsize(os.path.join(a.out, "phoneme.onnx")) // 1e6, "MB")
    import onnxruntime as ort, editdistance, time
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    src = os.path.join(a.out, "phoneme.onnx"); pre = os.path.join(a.out, "pre.onnx")
    try: quant_pre_process(src, pre, skip_optimization=False)
    except Exception as e: print("ön işleme atlandı:", e); pre = src
    variants = {
        "A_mm_lstm_pc": dict(op_types_to_quantize=["MatMul", "Gemm", "LSTM"], per_channel=True, weight_type=QuantType.QInt8),

    }
    ref = {}
    s0 = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    for c in cases:
        x = np.fromfile(os.path.join(a.audio, c["file"]), dtype=np.float32)
        ref[c["name"]] = greedy(torch.from_numpy(s0.run(None, {"wav": x[None]})[0][0]))
        print(f"   [fp32] {c['name']}: {' '.join(ref[c['name']])}")
    for vn, kw in variants.items():
        out = os.path.join(a.out, f"phoneme_{vn}.onnx")
        try:
            quantize_dynamic(pre, out, **kw)
        except Exception as e:
            print(vn, "nicemleme hatası:", e); continue
        sz = os.path.getsize(out) / 1e6
        s1 = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        errs = tot = 0; tsum = 0
        for c in cases:
            x = np.fromfile(os.path.join(a.audio, c["file"]), dtype=np.float32)
            t0 = time.time(); p = greedy(torch.from_numpy(s1.run(None, {"wav": x[None]})[0][0])); tsum += time.time() - t0
            errs += editdistance.eval(p, ref[c["name"]]); tot += len(ref[c["name"]])
            print(f"   [{vn}] {c['name']}: {' '.join(p)}")
        print(f"== {vn}: {sz:.0f} MB, fp32'ye göre fark %{100*errs/max(tot,1):.1f}, süre {tsum:.1f}s")
    json.dump({"vocab": vocab, "blank": getattr(tok, "pad_idx", 0)}, open(os.path.join(a.out, "vocab.json"), "w"), ensure_ascii=False)
    fin = os.path.join(a.out, "final"); os.makedirs(fin, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(a.out, "phoneme_A_mm_lstm_pc.onnx"), os.path.join(fin, "phoneme.onnx"))
    shutil.copy(os.path.join(a.out, "vocab.json"), os.path.join(fin, "vocab.json"))
    print("son model:", os.path.getsize(os.path.join(fin, "phoneme.onnx")) // 1e6, "MB")
except Exception as e:
    import traceback; traceback.print_exc()
