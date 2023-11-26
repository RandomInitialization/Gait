conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "your dataset path",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 74,
        'pid_shuffle': False},
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 8),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 0,
        'frame_num': 30,
        'model_name': 'GaitSet',
        'channel':[1,32,64,128]},
}
