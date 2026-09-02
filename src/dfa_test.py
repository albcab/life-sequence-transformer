import hydra
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from pathlib import Path
import logging
import numpy as np

# from life_sequence_dfa.dfa_idx import LIFESEQUENCEDFA
from life_sequence_dfa.new.dfa import LIFESEQUENCEDFA
dfa = LIFESEQUENCEDFA()

HOME_PATH = str(Path())

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="gconfig", version_base=None)
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    # print(cfg.ckpt_path)
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    
    data = instantiate(cfg.datamodule, _convert_="all")

    token2index = data.vocabulary.token2index
    index2token = data.vocabulary.index2token
    
    # data.setup(stage="test")
    # dataloader = data.test_dataloader()
    data.setup(stage="fit")
    dataloader = data.val_dataloader()

    eoy_idx = token2index['[EOY]']

    for batch in dataloader:

        names = batch['sequence_id'].detach().cpu().numpy()
        original_sequences = batch['original_sequence'].detach().cpu().numpy()
        knowns = batch['padding_mask'].detach().clone().cpu().bool()
        # original_tokens = np.vectorize(lambda i: index2token[i])(original_sequences)
        # for name, tokens, known in zip(names, original_tokens, knowns):
        for name, tokens, known in zip(names, original_sequences, knowns):
            print(name)
            #Extend to the last (since input is one shorter than total length)
            #Should be using decoder only datasets
            length = sum(known)
            known[length] = True

            path = dfa.get_initial_conf(sequence=tokens[known], verbose=False)

            if not path: #empty set
                original_tokens = np.vectorize(lambda i: index2token[i])(tokens)
                print(tokens[known])
                print(original_tokens[known])
                print("The input sequence is rejected by the DFA.")
                path = dfa.get_initial_conf(sequence=tokens[known], verbose=True)
            else:
                print("Last state of the input sequence in the DFA:", path)
                keys = [element[0] for element in path]
                # print("State reached:", key)
                if any([key in dfa.accepting_states for key in keys]):
                    print("The input sequence is accepted by the DFA.")
                else:
                    print("The input sequence is incomplete.")
            # input()


if __name__ == "__main__":
    main()