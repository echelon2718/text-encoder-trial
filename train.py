import argparse
import torch
from torch.utils.data import DataLoader, random_split

from data.dataset import AugmentDataset, collate_fn
from modules.tlejepa import TLeJEPA
from modules.training import Trainer
from modules.losses import (
    TLeJEPACriterion,
    SIGReg,
    SemanticTeacherSimilarity,
    CachedSemanticTeacherSimilarity,
)

def get_args():
    parser = argparse.ArgumentParser(description="Train TLeJEPA")
    parser.add_argument(
        "--lexicon-path",
        type=str,
        default="./lexicon/abbrev-lexicon.json"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../data/complete_corpus.csv"
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="huggingface",
        choices=["local", "huggingface"]
    )

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--num-slices", type=int, default=1024)
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--beta_", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main(args):
    dataset = AugmentDataset(
        lexicon_path=args.lexicon_path,
        dataset_path=args.dataset_path,
        mode=args.dataset_mode,
    )

    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = TLeJEPA(
        n_vocab_text=256,
        n_vocab_phoneme=dataset.phoneme_tokenizer.vocab_size,
        d_model=args.d_model,
        n_attn_heads=args.heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        max_length=args.max_length,
        dropout=args.dropout,
    )

    criterion = TLeJEPACriterion(
        sigreg_fn=SIGReg(num_slices=args.num_slices),
        cossim_fn=CachedSemanticTeacherSimilarity(
            SemanticTeacherSimilarity()
        ),
        lambda_=args.lambda_,
        beta_=args.beta_,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device(args.device),
    )

    trainer.fit(
        num_epochs=args.epochs,
        lr_scheduler=None,
    )

if __name__ == "__main__":
    args = get_args()
    main(args)
