
import hydra

from src.dataloaders.populations.users_puf import UserMonthPopulation
from src.dataloaders.sources.labor_puf import LaborMonthSource
from src.dataloaders.datamodule import Corpus
from src.dataloaders.vocabulary import CorpusVocabulary
from src.dataloaders.datamodule import L2VDataModule

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    
    users = UserMonthPopulation(name=cfg.datamodule.corpus.population.name)
    users.prepare()

    labor = LaborMonthSource(name=cfg.data_new.sources.labour_puf.name)
    labor.prepare()

    # from src.dataloaders.sources.health_puf import HealthMonthSource
    # health = HealthMonthSource(name=cfg.data_new.sources.health_puf.name)
    # health.prepare()

    sources = [
        labor,
        # health,
    ]

    corpus = Corpus(population=users, 
                    sources=sources, 
                    name=cfg.datamodule.corpus.name,
                    reference_year=cfg.datamodule.corpus.reference_year,
                    threshold_year=cfg.datamodule.corpus.threshold_year,
                    threshold_month=cfg.datamodule.corpus.threshold_month)
    corpus.prepare()

    vocab = CorpusVocabulary(corpus, name=cfg.datamodule.vocabulary.name, min_token_count=cfg.datamodule.vocabulary.min_token_count)
    vocab.prepare()

    if cfg.name == "pretraining_encoder" or cfg.name == "pretraining_performer":
        from src.dataloaders.tasks.pretrain import MLM
        task = MLM(name=cfg.datamodule.task.name, 
                    max_length=cfg.datamodule.task.max_length, 
                    p_sequence_reverse_events=cfg.datamodule.task.p_sequence_reverse_events,
                    p_sequence_shuffle_events=cfg.datamodule.task.p_sequence_shuffle_events,
                    p_sequence_shuffle_tokens=cfg.datamodule.task.p_sequence_shuffle_tokens,
                    p_sentence_drop_tokens=cfg.datamodule.task.p_sentence_drop_tokens,
                    mask_ratio=cfg.datamodule.task.mask_ratio)
    
    if cfg.name == "pretraining_decoder":
        from src.dataloaders.tasks.decode import Decode
        task = Decode(name=cfg.datamodule.task.name, 
                    max_length=cfg.datamodule.task.max_length,
                    p_sequence_reverse_events=cfg.datamodule.task.p_sequence_reverse_events,
                    p_sequence_shuffle_events=cfg.datamodule.task.p_sequence_shuffle_events,
                    p_sequence_shuffle_tokens=cfg.datamodule.task.p_sequence_shuffle_tokens,
                    p_sentence_drop_tokens=cfg.datamodule.task.p_sentence_drop_tokens)
        
    if cfg.name == "decoder_only":
        from src.dataloaders.tasks.decode_only import DecodeOnly
        task = DecodeOnly(name=cfg.datamodule.task.name,
                    max_length=cfg.datamodule.task.max_length,
                    p_sequence_reverse_events=cfg.datamodule.task.p_sequence_reverse_events,
                    p_sequence_shuffle_events=cfg.datamodule.task.p_sequence_shuffle_events,
                    p_sequence_shuffle_tokens=cfg.datamodule.task.p_sequence_shuffle_tokens,
                    p_sentence_drop_tokens=cfg.datamodule.task.p_sentence_drop_tokens)
        
    datamodule = L2VDataModule(corpus, task=task, vocabulary=vocab,
                               batch_size=cfg.datamodule.batch_size,
                               num_workers=cfg.datamodule.num_workers)
    datamodule.prepare()








    dataloader = datamodule.train_dataloader()
    sequence_id_sum = 0
    sequence_id_count = 0
    seq_length = []
    n = 0

    for batch in dataloader:
        # sequence_ids = batch["sequence_id"]
        # batch_size = sequence_ids.shape[0]
        # n += batch_size
        padding_mask = batch["padding_mask"]
        # sequence_id_sum += sequence_ids.sum().item()
        # sequence_id_count += sequence_ids.numel()  # Number of elements in sequence_ids
        length = padding_mask.sum(axis=1)
        # print(length, batch["sequence_id"])
        seq_length.append(length)

    import torch
    seq_length = torch.concat(seq_length)

    print(torch.mean(seq_length, dtype=float))
    print(torch.median(seq_length))
    print(torch.mode(seq_length))
    print(torch.max(seq_length))
    print(torch.min(seq_length))

    for i, b in enumerate(torch.bincount(seq_length.to(int))):
        print(i, *["." for a in range(int(b))], sep="")






if __name__ == "__main__":
    main()