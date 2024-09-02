import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/PTB.yaml")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--detail", type=str, default='')
    parser.add_argument("--clip_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    return parser, args