# -*- coding: utf-8 -*-
"""Torch heads and the shared training loop.

``DualHead`` is a verbatim functional copy of the head in
``20260929_235340_FusionAuxNorm/code/deep_cv_expanded.py`` (the published
DNABERT-2 run).  ``ProteinOnlyHead`` keeps the same protein normalisation,
masked view attention, projection, L2 scale normalisation, classifier and
protein auxiliary head, but removes the DNA branch, the gate and the
reverse-complement consistency term.

Keeping both heads in one file makes the matched comparison auditable: the
only differences are the DNA pathway and the loss terms that depend on it.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset


def masked_softmax(scores, mask, dim):
    scores = scores.masked_fill(~mask, -1e4)
    w = torch.softmax(scores, dim=dim) * mask.to(scores.dtype)
    return w / w.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class DualHead(nn.Module):
    """Published DNA+protein normalized-gate head (DNABERT-2 configuration)."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        dna_dim = int(m["dna_hidden_size"])
        prot_dim = int(m["protein_hidden_size"])
        gru = int(m["bigru_hidden_size"])
        fd = int(m["fusion_dim"])
        attn = int(m["attention_hidden_size"])
        drop = float(m["dropout"])
        self.fusion_dim = fd
        self.dna_input_norm = nn.LayerNorm(dna_dim)
        self.bigru = nn.GRU(dna_dim, gru, batch_first=True, bidirectional=True)
        self.dna_view_score = nn.Sequential(nn.Linear(2 * gru, attn), nn.Tanh(), nn.Linear(attn, 1))
        self.strand_project = nn.Sequential(nn.Linear(2 * gru, fd), nn.GELU())
        self.channel_score = nn.Sequential(nn.Linear(fd, attn), nn.Tanh(), nn.Linear(attn, 1))
        self.protein_input_norm = nn.LayerNorm(prot_dim)
        self.protein_view_score = nn.Sequential(nn.Linear(prot_dim, attn), nn.Tanh(), nn.Linear(attn, 1))
        self.protein_project = nn.Sequential(nn.Linear(prot_dim, fd), nn.GELU())
        self.dna_fusion_norm = nn.LayerNorm(fd)
        self.protein_fusion_norm = nn.LayerNorm(fd)
        gh = max(64, fd // 2)
        self.gate = nn.Sequential(nn.Linear(4 * fd + 1, gh), nn.GELU(), nn.Dropout(drop / 2), nn.Linear(gh, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.fused_norm = nn.LayerNorm(fd)
        self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, fd // 2), nn.GELU(),
                                        nn.Dropout(drop), nn.Linear(fd // 2, 1))
        self.dna_aux = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, 1))
        self.prot_aux = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, 1))

    def _eq(self, v, n):
        v = n(v)
        return F.normalize(v, p=2, dim=-1, eps=1e-8) * math.sqrt(self.fusion_dim)

    def forward(self, dna, dm, prot, pm):
        b, s, v, _ = dna.shape
        dna = self.dna_input_norm(dna).reshape(b * s, v, -1)
        rm = dm[:, None, :].expand(b, s, v).reshape(b * s, v)
        go, _ = self.bigru(dna)
        dvw = masked_softmax(self.dna_view_score(go).squeeze(-1), rm.bool(), 1)
        strand = torch.sum(dvw.unsqueeze(-1) * go, 1)
        strand = self.strand_project(strand).reshape(b, s, -1)
        cw = masked_softmax(self.channel_score(strand).squeeze(-1),
                            torch.ones((b, s), dtype=torch.bool, device=dna.device), 1)
        hd = torch.sum(cw.unsqueeze(-1) * strand, 1)
        hd = self._eq(hd, self.dna_fusion_norm)
        prot = self.protein_input_norm(prot)
        pm = pm.bool()
        pvw = masked_softmax(self.protein_view_score(prot).squeeze(-1), pm, 1)
        ps = torch.sum(pvw.unsqueeze(-1) * prot, 1)
        hp = self.protein_project(ps)
        avail = pm.any(1)
        hp = self._eq(hp, self.protein_fusion_norm) * avail.unsqueeze(-1).to(hp.dtype)
        gi = torch.cat([hd, hp, torch.abs(hd - hp), hd * hp,
                        avail.to(hd.dtype).unsqueeze(-1)], -1)
        g = torch.sigmoid(self.gate(gi))
        g = torch.where(avail.unsqueeze(-1), g, torch.ones_like(g))
        fused = self.fused_norm(g * hd + (1 - g) * hp)
        rc = (1.0 - F.cosine_similarity(strand[:, 0], strand[:, 1], -1)).mean()
        return {"logits": self.classifier(fused).squeeze(-1),
                "dna_aux": self.dna_aux(hd).squeeze(-1),
                "prot_aux": self.prot_aux(hp).squeeze(-1),
                "avail": avail, "rc": rc}


class ProteinOnlyHead(nn.Module):
    """Protein-only counterpart with the same protein pathway and classifier."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        prot_dim = int(m["protein_hidden_size"])
        fd = int(m["fusion_dim"])
        attn = int(m["attention_hidden_size"])
        drop = float(m["dropout"])
        self.fusion_dim = fd
        self.protein_input_norm = nn.LayerNorm(prot_dim)
        self.protein_view_score = nn.Sequential(nn.Linear(prot_dim, attn), nn.Tanh(), nn.Linear(attn, 1))
        self.protein_project = nn.Sequential(nn.Linear(prot_dim, fd), nn.GELU())
        self.protein_fusion_norm = nn.LayerNorm(fd)
        self.classifier = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, fd // 2), nn.GELU(),
                                        nn.Dropout(drop), nn.Linear(fd // 2, 1))
        self.prot_aux = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, 1))

    def _eq(self, v, n):
        v = n(v)
        return F.normalize(v, p=2, dim=-1, eps=1e-8) * math.sqrt(self.fusion_dim)

    def forward(self, dna, dm, prot, pm):
        prot = self.protein_input_norm(prot)
        pm = pm.bool()
        pvw = masked_softmax(self.protein_view_score(prot).squeeze(-1), pm, 1)
        ps = torch.sum(pvw.unsqueeze(-1) * prot, 1)
        hp = self.protein_project(ps)
        avail = pm.any(1)
        hp = self._eq(hp, self.protein_fusion_norm) * avail.unsqueeze(-1).to(hp.dtype)
        return {"logits": self.classifier(hp).squeeze(-1),
                "prot_aux": self.prot_aux(hp).squeeze(-1),
                "avail": avail}


class DS(Dataset):
    def __init__(self, idx, dna, dm, prot, pm, labels):
        self.idx = np.asarray(idx, dtype=np.int64)
        self.dna, self.dm, self.prot, self.pm, self.labels = dna, dm, prot, pm, labels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return {"dna": torch.from_numpy(self.dna[j].astype(np.float32)),
                "dm": torch.from_numpy(np.asarray(self.dm[j]).copy()),
                "prot": torch.from_numpy(self.prot[j].astype(np.float32)),
                "pm": torch.from_numpy(np.asarray(self.pm[j]).copy()),
                "label": torch.tensor(float(self.labels[j]), dtype=torch.float32)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_thr(y, p, frozen):
    best = (None, -float("inf"))
    for t in np.arange(float(frozen["threshold_min"]), float(frozen["threshold_max"]) + 1e-9,
                       float(frozen["threshold_step"])):
        mcc = float(matthews_corrcoef(y, (p >= t).astype(int)))
        if mcc > best[1]:
            best = (t, mcc)
    return best[0]


def predict(model, loader, device):
    model.eval()
    py, yy = [], []
    with torch.no_grad():
        for b in loader:
            o = model(b["dna"].to(device), b["dm"].to(device),
                      b["prot"].to(device), b["pm"].to(device))
            py.extend(torch.sigmoid(o["logits"]).cpu().numpy().tolist())
            yy.extend(b["label"].numpy().tolist())
    return np.asarray(yy), np.asarray(py)


def train_fold(model_fn, arrays, labels, tr, va, te, seed, cfg, device, checkpoint_path=None):
    seed_everything(seed)
    model = model_fn().to(device)
    frozen = cfg["frozen_training"]

    def loader(idx, shuf):
        ds = DS(idx, arrays["dna"], arrays["dm"], arrays["prot"], arrays["pm"], labels)
        g = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=int(frozen["batch_size"]), shuffle=shuf, generator=g)

    tr_ld, va_ld, te_ld = loader(tr, True), loader(va, False), loader(te, False)
    n_pos = int(labels[tr].sum())
    n_neg = int(len(tr) - n_pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(1, n_pos)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(frozen["learning_rate"]),
                            weight_decay=float(frozen["weight_decay"]))
    dw = float(cfg["improvement"]["dna_aux_loss_weight"])
    pw = float(cfg["improvement"]["protein_aux_loss_weight"])
    rw = float(frozen["rc_consistency_weight"])
    best, stale, state, best_epoch = -float("inf"), 0, None, 0
    for ep in range(1, int(frozen["epochs"]) + 1):
        model.train()
        for b in tr_ld:
            opt.zero_grad(set_to_none=True)
            o = model(b["dna"].to(device), b["dm"].to(device),
                      b["prot"].to(device), b["pm"].to(device))
            lab = b["label"].to(device)
            loss = crit(o["logits"], lab)
            if "dna_aux" in o:
                loss = loss + dw * crit(o["dna_aux"], lab)
            if "prot_aux" in o:
                av = o["avail"]
                if bool(av.any()):
                    loss = loss + pw * crit(o["prot_aux"][av], lab[av])
            if "rc" in o:
                loss = loss + rw * o["rc"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(frozen["gradient_clip_norm"]))
            opt.step()
        vy, vp = predict(model, va_ld, device)
        vauc = float(roc_auc_score(vy, vp))
        if vauc > best + 1e-6:
            best, stale, best_epoch = vauc, 0, ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(frozen["patience"]):
                break
    model.load_state_dict(state)
    if checkpoint_path is not None:
        torch.save({"state_dict": state, "seed": seed, "best_epoch": best_epoch, "best_validation_auroc": best}, checkpoint_path)
    vy, vp = predict(model, va_ld, device)
    thr = choose_thr(vy, vp, frozen)
    ty, tp = predict(model, te_ld, device)
    return ty, tp, float(thr), int(best_epoch), float(best)
