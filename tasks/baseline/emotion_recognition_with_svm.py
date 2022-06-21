from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from src.utils.dataset import SeedDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import sklearn.metrics as metrics
from omegaconf import OmegaConf


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    # np.random.seed(cfg.basic.seed)
    data_path = to_absolute_path(cfg.basic.data_path)
    dataset = SeedDataset(
        data_path=data_path,
        target_subject=cfg.basic.target_sub,
        do_quantization=cfg.basic.do_quantization,
        quantization_level=cfg.basic.quantization_level,
        do_normalization=cfg.basic.do_normalization,
    )
    model = None

    if cfg.basic.model == 'SVC':
        model = SVC(
            C=cfg.model.SVC.C,
            kernel=cfg.model.SVC.kernel,
            degree=cfg.model.SVC.degree,
            tol=cfg.model.SVC.tol,
            cache_size=cfg.model.SVC.cache_size,
            verbose=cfg.model.SVC.verbose,
            gamma=cfg.model.SVC.gamma,
            decision_function_shape=cfg.model.SVC.decision_function_shape,
        )
    elif cfg.basic.model == 'LR':
        model = LogisticRegression(
            C=cfg.model.LR.C,
            tol=cfg.model.LR.tol,
            solver=cfg.model.LR.solver,
            max_iter=cfg.model.LR.max_iter,
            multi_class=cfg.model.LR.multi_class,
            verbose=cfg.model.LR.verbose,
            n_jobs=cfg.model.LR.n_cpu,
        )
    elif cfg.basic.model == 'AdaBoostClassifier':
        model = AdaBoostClassifier(
            n_estimators=cfg.model.ABC.n_estimators,
            learning_rate=cfg.model.ABC.learning_rate,
            algorithm=cfg.model.ABC.algorithm,
            random_state=cfg.basic.seed,
        )
    elif cfg.basic.model == 'LinearSVC':
        model = LinearSVC(
            C=cfg.model.LSVC.C,
            tol=cfg.model.LSVC.tol,
            loss=cfg.model.LSVC.loss,
            multi_class=cfg.model.LSVC.multi_class,
            verbose=cfg.model.LSVC.verbose,
            max_iter=cfg.model.LSVC.max_iter,
        )

    if cfg.basic.do_quantization:
        source_data = dataset.quantized_source_data.numpy() # (3394*4, 310)
        source_labels = dataset.source_class_labels.numpy() # (3397*4, )
        target_data = dataset.quantized_target_data.numpy()  # (3394, 310)
        target_labels = dataset.target_class_labels.numpy() # (3394, )
    else:
        source_data = dataset.source_data.numpy() # (3394*4, 310)
        source_labels = dataset.source_class_labels.numpy() # (3397*4, )
        target_data = dataset.target_data.numpy()  # (3394, 310)
        target_labels = dataset.target_class_labels.numpy() # (3394, )

    scaler = StandardScaler()
    scaler.fit(source_data)
    source_data = scaler.transform(source_data)
    scaler.fit(target_data)
    target_data = scaler.transform(target_data)

    model.fit(X=source_data, y=source_labels)
    # predictions = model.predict(X=target_data)
    accuracy = model.score(target_data, target_labels)
    with open(to_absolute_path(cfg.basic.log_file), 'a') as f:
        if cfg.basic.model == 'SVC':
            f.write(f"{cfg.basic.model},{cfg.basic.target_sub},{cfg.model.SVC.C},{cfg.model.SVC.kernel},{cfg.model.SVC.tol},{accuracy}\n")
        elif cfg.basic.model == 'LR':
            f.write(f"{cfg.basic.model},{cfg.basic.target_sub},{cfg.model.LR.C},{cfg.model.LR.solver},{cfg.model.LR.tol},{accuracy}\n")
        elif cfg.basic.model == 'AdaBoostClassifier':
            f.write(f"{cfg.basic.model},{cfg.basic.target_sub},{cfg.model.ABC.n_estimators},{cfg.model.ABC.learning_rate},{cfg.model.ABC.algorithm},{accuracy}\n")
        elif cfg.basic.model == 'LinearSVC':
            f.write(f"{cfg.basic.model},{cfg.basic.target_sub},{cfg.model.LSVC.C},{cfg.model.LSVC.tol},{cfg.model.LSVC.loss},{accuracy}\n")
    print(f"{cfg.basic.model},{cfg.basic.target_sub},{accuracy}")


if __name__ == '__main__':
    run()
