import os, random, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
 
DATA_DIR       = '/kaggle/input/datasets/divya21yadav/facial-emotion-dataset/facial_emotion_dataset_4'
CHECKPOINT_DIR = '/kaggle/working/'
MODEL_PATH     = CHECKPOINT_DIR + 'best_emotion_model.keras'
PHASE_LOG_PATH = CHECKPOINT_DIR + 'training_phase.json'
 
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 4
 
print("="*60)
print("Classes found:", os.listdir(DATA_DIR))
print("="*60)
def save_phase(phase_num, best_val_acc):
    with open(PHASE_LOG_PATH, 'w') as f:
        json.dump({'completed_phase': phase_num, 'best_val_acc': float(best_val_acc)}, f)
    print(f"\n✓ Phase {phase_num} saved. Best val_acc so far: {best_val_acc:.4f}")
 
def load_phase():
    if os.path.exists(PHASE_LOG_PATH):
        with open(PHASE_LOG_PATH) as f:
            data = json.load(f)
        print(f"✓ Resuming — last completed phase: {data['completed_phase']} | Best acc: {data['best_val_acc']:.4f}")
        return data['completed_phase'], data['best_val_acc']
    print("No checkpoint found — starting fresh from Phase 1.")
    return 0, 0.0

train_gen = ImageDataGenerator(
    # rescale = 1./255   <-- REMOVED, this was the bug
    rotation_range     = 20,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    zoom_range         = 0.15,
    shear_range        = 0.1,
    horizontal_flip    = True,
    brightness_range   = [0.75, 1.25],
    fill_mode          = 'nearest',
    validation_split   = 0.2
)
 
val_gen = ImageDataGenerator(
    # rescale = 1./255   <-- REMOVED
    validation_split = 0.2
)
 
train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'training',
    shuffle     = True,
    seed        = SEED
)
 
val_data = val_gen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    subset      = 'validation',
    shuffle     = False,
    seed        = SEED
)
 
print(f"\nClass indices : {train_data.class_indices}")
print(f"Train samples : {train_data.samples}")
print(f"Val samples   : {val_data.samples}")

classes           = train_data.classes
cw                = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weight_dict = dict(zip(np.unique(classes), cw))
print(f"\nClass weights : {class_weight_dict}")
def build_model(num_classes=4, img_size=224, freeze_base=True):
    base = EfficientNetV2S(
        weights             = 'imagenet',
        include_top         = False,
        include_preprocessing = True,   # model handles its own rescaling
        input_shape         = (img_size, img_size, 3)
    )
    base.trainable = not freeze_base
 
    inputs = Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
 
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
 
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
 
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs), base
  callbacks = [
    EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 6,
        restore_best_weights = True,
        verbose              = 1
    ),
    ModelCheckpoint(
        filepath       = MODEL_PATH,
        monitor        = 'val_accuracy',
        save_best_only = True,
        mode           = 'max',
        verbose        = 1
    ),
    ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.3,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1
    )
]
completed_phase, best_val_acc = load_phase()
 
if completed_phase >= 1 and os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = model.layers[1]
    print("✓ Model loaded from checkpoint.")
else:
    model, base_model = build_model(NUM_CLASSES, IMG_SIZE, freeze_base=True)
    print("✓ Fresh model built.")
 
model.summary(line_length=80)

if completed_phase < 1:
    print("\n" + "="*60)
    print("PHASE 1: Training classification head (base frozen)")
    print("="*60)
 
    base_model.trainable = False
 
    model.compile(
        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics   = ['accuracy']
    )
 
    history1 = model.fit(
        train_data,
        validation_data = val_data,
        epochs          = 10,
        callbacks       = callbacks,
        class_weight    = class_weight_dict
    )
 
    best_val_acc = max(history1.history['val_accuracy'])
    save_phase(1, best_val_acc)
 
else:
    print("\n✓ PHASE 1 already completed — skipping.")
    history1 = None
  if completed_phase < 2:
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning top 60 layers")
    print("="*60)
 
    base_model = model.layers[1]
    base_model.trainable = True
 
    UNFREEZE_FROM = len(base_model.layers) - 60
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= UNFREEZE_FROM)
 
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfrozen {trainable_count} / {len(base_model.layers)} layers")
 
    model.compile(
        optimizer = AdamW(learning_rate=5e-5, weight_decay=1e-4),
        loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics   = ['accuracy']
    )
 
    history2 = model.fit(
        train_data,
        validation_data = val_data,
        epochs          = 25,
        callbacks       = callbacks,
        class_weight    = class_weight_dict
    )
 
    best_val_acc = max(max(history2.history['val_accuracy']), best_val_acc)
    save_phase(2, best_val_acc)
 
else:
    print("\n✓ PHASE 2 already completed — skipping.")
    history2 = None
  if completed_phase < 3 and best_val_acc < 0.80:
    print("\n" + "="*60)
    print(f"PHASE 3: Full fine-tune — current best: {best_val_acc:.4f}")
    print("="*60)
 
    base_model = model.layers[1]
    base_model.trainable = True
 
    model.compile(
        optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics   = ['accuracy']
    )
 
    history3 = model.fit(
        train_data,
        validation_data = val_data,
        epochs          = 15,
        callbacks       = callbacks,
        class_weight    = class_weight_dict
    )
 
    best_val_acc = max(max(history3.history['val_accuracy']), best_val_acc)
    save_phase(3, best_val_acc)
 
elif best_val_acc >= 0.80:
    print(f"\n✓ PHASE 3 skipped — target met! (val_acc = {best_val_acc:.4f})")
    history3 = None
else:
    print("\n✓ PHASE 3 already completed — skipping.")
    history3 = None
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
 
best_model  = tf.keras.models.load_model(MODEL_PATH)
val_data.reset()
 
y_pred_probs = best_model.predict(val_data, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = val_data.classes[:len(y_pred)]
class_names  = list(val_data.class_indices.keys())
 
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))
 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — Best Model', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(CHECKPOINT_DIR + 'confusion_matrix.png', dpi=150)
plt.show()
