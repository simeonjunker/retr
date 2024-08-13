from os.path import join


class Config(object):
    def __init__(self):

        self.prefix = 'refcoco'

        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 20
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'ResNet152'
        self.position_embedding = 'sine'
        self.dilation = True
        
        # Basic
        self.device = 'cuda'
        self.save_every = 'max'
        self.seed = 42
        self.batch_size = 32
        self.num_workers = 8
        self.checkpoint = f'./{self.prefix}_checkpoint.pth'
        self.project_data_path = './data'
        self.checkpoint_path = join(self.project_data_path, 'models', self.prefix)
        self.clip_max_norm = 0.1
        self.early_stopping = True
        self.stop_after_epochs = 3
        self.use_global_features = False
        self.use_location_features = False
        self.use_scene_summaries = False
        self.scene_summary_type='annotated'  # 'annotated' or 'predicted'
        self.verbose = True

        # Transformer
        self.transformer_type = 'Concat'
        self.hidden_dim = 512
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 3
        self.dec_layers = 3
        self.dim_feedforward = 512
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = 'PATH_TO_COCO'  # COCO base dir (images)
        self.ref_base = 'PATH_TO_REF_BASE'  # RefCOCO* base dir (annotations)
        self.ref_dir = join(self.ref_base, self.prefix)
        self.limit = -1