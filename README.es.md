# PyCoH

*[English](README.md) · Español*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22972086.svg)](https://doi.org/10.5281/zenodo.22972086)

Adaptador de ajuste fino eficiente en parámetros para transformers. Inyecta
una corrección direccional en el residual stream de cada bloque, con el
modelo base completamente congelado.

```python
from transformers import AutoModelForCausalLM
from pycoh import apply_coh, remove_coh, save_adapter, load_adapter

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M").cuda()
apply_coh(model, d_tau=96)

# ...tu bucle de entrenamiento habitual...

save_adapter(model, "mi_adaptador.pt")
```

Para reutilizarlo sobre una instancia limpia del mismo modelo base:

```python
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M").cuda()
apply_coh(model, d_tau=96)
load_adapter(model, "mi_adaptador.pt")
```

El ciclo de vida completo es simétrico:

```
apply_coh     instalar el adaptador
remove_coh    desinstalarlo y restaurar la topología
save_adapter  persistirlo
load_adapter  restaurarlo
```

---

## Qué hace

Para el estado oculto `h` a la entrada de cada bloque:

```
z  = W_τ h                    comprime a d_τ dimensiones
r  = min(σ(φ(z)), r_max)      compuerta por token, con techo
s  = 1/√(1 − r²) − 1          amplitud
Ĵ  = normalize(W_o z)         dirección, de norma unitaria
Δh = β · Ĵ · s
h' = B(h) + Δh
```

La corrección se calcula sobre la **entrada** del bloque, no sobre su
salida. La dirección depende del estado actual; su norma es siempre 1, de
modo que la amplitud queda desacoplada de la magnitud del estado. El techo
`r_max` acota `s` a 4.0252 e impide la divergencia de `1/√(1−r²)`.

Parámetros entrenables por capa: `2·d_model·d_τ + d_τ`.

## Números medidos

Sobre `SmolLM2-360M` con `d_tau=96`, 32 capas:

| | |
|---|---|
| Parámetros del adaptador | 5 901 344 (1.6% del modelo) |
| Entrenables por defecto | 5 901 312 |
| Tamaño del archivo | 23.6 MB (frente a 724 MB del modelo) |
| Modelo base modificado | 0 tensores, verificado tras entrenar |

Verificado en ejecución sobre T4: `Trainer` de HuggingFace, base en
`bfloat16`, gradient checkpointing activo, guardado y recarga del adaptador
sobre un modelo recién descargado reproduciendo la pérdida.

**El repositorio no declara rendimiento agregado del modelo de lenguaje.**
Un estudio separado y preregistrado, P5-M, reporta una comparación
confirmatoria de CoH con la configuración LoReFT 2-prefix/2-suffix sobre
SmolLM2-360M-Instruct congelado. Su endpoint es la persistencia posicional:
la pendiente con la posición relativa de la mejora de pérdida por token,
no la pérdida agregada, perplexity, utilidad downstream ni calidad general
del modelo. El estudio obtuvo ΔD > 0 en 48 de 69 textos (prueba exacta
unilateral de signos, p = 0.00078; media ΔD = 0.0948; mediana = 0.0720).
El resultado está restringido al backbone, corpus, longitud de secuencia,
protocolo de entrenamiento y configuraciones de intervención probados; no
es una afirmación de superioridad general.

Artículo: **P5-M — Positional Persistence of a Dynamic Activation Correction**
(DOI: `10.5281/zenodo.22865398`), preregistrado en
`10.17605/OSF.IO/X5W8T`.

## Precisión numérica

Toda la aritmética del núcleo corre en FP32, y el resultado se devuelve en
el dtype de entrada. Esto se mantiene **bajo precisión mixta sin que haya
que configurar nada**: `torch.autocast` convierte las entradas de las
operaciones lineales a precisión baja sin importar el dtype que se les
pase, así que el núcleo desactiva autocast internamente. Un modelo base en
`bfloat16` con un adaptador en FP32 es la configuración normal y funciona
sola.

## Limitaciones

Están documentadas porque son propiedades del diseño, no defectos por
corregir.

**No se puede fusionar en los pesos.** `Δh` es función de la activación,
no un delta de pesos, así que no existe `merge_and_unload()`. A diferencia
de LoRA, el costo se paga en cada forward. El sobrecosto de latencia
todavía no se ha medido y no se declara.

**`apply_coh()` no es la identidad al inicializar.** `out_proj` arranca
con el init por defecto de PyTorch, así que la corrección es no nula desde
el primer forward. En SmolLM2-360M eso produce un cambio máximo en los
logits de ~0.65 antes de entrenar. Quien venga de LoRA espera lo contrario.

Inicializar `out_proj` a cero **no** es la solución: `F.normalize` divide
por `‖x‖.clamp_min(eps)`, de modo que en el origen la derivada vale
`1/eps ≈ 1e12`. Medido: norma de gradiente `2.18e12` en cuanto el gradiente
que llega desde arriba es no nulo. Conseguir identidad al inicio exige otro
diseño y queda para una versión posterior.

**`save_pretrained()` no funciona mientras CoH está instalado.** Al
envolver cada bloque, las claves pasan de `model.layers.3.self_attn...` a
`model.layers.3.block.self_attn...`, así que `from_pretrained` ya no puede
reconstruirlo. Por eso el adaptador se guarda aparte con `save_adapter`. Si
necesitás el checkpoint base, `remove_coh(model)` restaura la topología
original y `save_pretrained` vuelve a funcionar.

**`d_tau` es obligatorio y no tiene valor por defecto.** La única
proporción con respaldo empírico es `d_tau/d_model = 0.1`, medida en
`d_model=960`. Extrapolarla a otros tamaños es una hipótesis, no una regla,
y la librería no la aplica en silencio.

**`trainable_beta=True` y weight decay.** Si activás `beta` como
parámetro entrenable y lo metés en un grupo de AdamW con `weight_decay`,
`beta` recibe decaimiento. Medido en una tarea de juguete a 150 pasos:
`0.5 → 1.0255` sin weight decay, `0.5 → 0.9743` con el 0.01 por defecto.
Es una deriva modesta, no un colapso, pero en entrenamientos largos
conviene darle a `beta` su propio grupo con `weight_decay=0.0`.

## API

### `apply_coh(model, *, d_tau, ...)`

Aplica CoH **in-place** y devuelve el mismo objeto. Descubre la pila de
bloques por estructura, no por nombres: busca `nn.ModuleList` homogéneos,
descarta los anidados dentro de otros candidatos y verifica que los
parámetros del bloque operen sobre `d_model`. **Si hay ambigüedad, falla**
y enumera los candidatos; nunca elige por su cuenta.

| Argumento | Por defecto | |
|---|---|---|
| `d_tau` | — | obligatorio |
| `layers` | `None` | `None`=todas, `N`=las primeras N, `[i,j]`=índices |
| `hidden_size` | `None` | override; si no, `config.hidden_size` o `config.d_model` |
| `target_modules` | `None` | ruta exacta del contenedor, para resolver ambigüedad |
| `beta_init` | `0.5` | |
| `r_max` | `0.98` | |
| `trainable_beta` | `False` | |
| `freeze` | `True` | congela todo lo que no sea CoH |

Cada adaptador se crea en el dispositivo de su bloque, así que un modelo
repartido entre varias GPUs funciona sin ajustes. El dtype no se hereda:
CoH permanece en FP32.

Una segunda llamada sobre un modelo ya inyectado lanza `RuntimeError`. Ante
cualquier error de validación el modelo queda **intacto**: se construyen
todos los wrappers antes de colocar el primero.

### `remove_coh(model, *, unfreeze=False)`

Desinstala CoH y restaura la topología original, in-place. Cada wrapper se
sustituye por el mismo objeto de bloque que envolvía, de modo que
`named_modules()` y `state_dict()` vuelven a ser exactamente los del modelo
limpio.

No restaura pesos: el modelo base nunca se modificó. El estado de CoH se
pierde salvo que se haya guardado antes con `save_adapter`.

`requires_grad` no se toca: `apply_coh` congeló la base y aquí no se sabe
cuál era el estado previo, así que adivinarlo sería peor que dejarlo
explícito. `unfreeze=True` reactiva todos los parámetros restantes.

Sobre un modelo sin CoH lanza `RuntimeError`.

### `save_adapter(model, path)` / `load_adapter(model, path)`

El archivo contiene exclusivamente el estado de los módulos CoH —ni un solo
peso del huésped— más metadatos: versión de formato, mecanismo, `d_model`,
`d_tau`, `r_max`, `trainable_beta` y la lista de rutas inyectadas.

`load_adapter` exige que el modelo **ya** tenga CoH aplicado; no modifica la
topología. Valida mecanismo, versión, rutas, `d_model`, `d_tau`, `r_max`,
claves y formas **antes** de escribir un solo tensor, así que un adaptador
incompatible no deja el modelo a medio cargar. Nunca usa `strict=False`.

`r_max` se valida explícitamente porque no altera ninguna forma: sin esa
comprobación, un adaptador entrenado con otro techo se cargaría sin error y
el mecanismo se comportaría distinto en silencio.

La deserialización usa `weights_only=True`. Los checkpoints de PyTorch
ejecutan código al abrirse; un adaptador descargado de internet no debería
poder hacerlo.

`trainable_beta` en el archivo es informativo: describe cómo se entrenó el
adaptador y no cambia la configuración del modelo receptor. El valor de
`beta`, en cambio, sí se restaura desde el archivo.

## Compatibilidad

Requiere que el modelo tenga una lista homogénea de bloques, que esos
bloques reciban el estado como primer argumento y devuelvan un `Tensor` o
una `tuple`, y que declare su dimensión en el config.

Verificado sobre `SmolLM2-360M` con `transformers 5.16`, `torch 2.11` y
`2.14`. Los modelos codificador-decodificador tienen dos pilas y exigen
`target_modules` explícito. Los modelos que declaran la dimensión como
`n_embd` (familia GPT-2) requieren pasar `hidden_size` a mano.

Una salida de bloque que no sea `Tensor` ni `tuple` produce `TypeError`
explícito: la librería no adivina cuál campo de un `dict` es el estado
oculto.

## Instalación

```bash
pip install pycoh
```

Desde el repositorio:

```bash
git clone <repo> && cd pycoh
pip install -e ".[dev]"
pytest -q
```

## Tests

```bash
pytest -q                                          # suite completa
python tests/diagnostics/zero_init_probe.py        # diagnóstico
PYTHONPATH=. python tests/integration/smollm2_run.py    # requiere red
PYTHONPATH=. python tests/integration/smollm2_train.py  # requiere red y GPU
```

Los tests del núcleo comparan cada etapa del cálculo **bit a bit** contra
una referencia escrita desde la especificación, que no llama al código bajo
prueba.
