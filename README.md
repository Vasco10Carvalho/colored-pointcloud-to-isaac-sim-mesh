# Colored Point Cloud to Isaac Sim Mesh

Pipeline reprodutível para transformar uma ou várias point clouds PLY com RGB
num ambiente portátil para NVIDIA Isaac Sim.

O resultado contém:

- mesh visual detalhada com cores por vértice;
- alternativa visual em pontos;
- mesh de colisão independente e simplificada;
- plano de colisão de segurança;
- stage USD em metros e Z-up;
- relatórios, métricas, validação e pré-visualizações;
- apenas referências relativas entre os ficheiros USD.

O PLY original nunca é modificado. Este repositório não necessita de ficheiros
`.env`, tokens ou caminhos específicos de um computador.

## Resultado final

Depois de executar o pipeline, abrir no Isaac Sim:

```text
output_environment/environment_ready.usda
```

Para visualizar os pontos RGB em vez da superfície reconstruída:

```text
output_environment/environment_points_ready.usda
```

## Requisitos

- Linux com Python 3.10 ou superior;
- NVIDIA Isaac Sim instalado apenas no computador onde o resultado será aberto;
- espaço livre suficiente para a point cloud e meshes intermédias;
- PLY com propriedades XYZ e RGB;
- coordenadas em metros e, por omissão, eixo vertical Z.

Instalar as dependências do processamento:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A pasta `.venv` é local e está ignorada pelo Git. Não é utilizado nenhum
ficheiro `.env`.

## Verificar o PLY antes de processar

```bash
python scripts/inspect_ply.py "./data/map.ply"
```

O relatório apresentado no terminal inclui número de pontos, XYZ/RGB,
bounding box, distância ao vizinho mais próximo, indícios sobre escala e o
eixo mais alinhado com o maior plano detetado.

Confirma sempre a escala comparando uma distância conhecida. A inferência
automática não consegue distinguir com certeza metros de centímetros sem uma
referência física.

## Execução automática

### Qualquer PLY colorido

Sem perfil, os voxels e profundidades Poisson são escolhidos a partir da
densidade medida:

```bash
python scripts/run_pipeline.py \
  "./data/map.ply" \
  "./output_environment"
```

Também é possível indicar explicitamente o perfil automático:

```bash
python scripts/run_pipeline.py \
  "./data/map.ply" \
  "./output_environment" \
  --profile configs/auto.json
```

Os caminhos são argumentos do comando. Não é preciso editar scripts nem
guardar caminhos num `.env`. A exportação usa `usd-core` e não inicia o
renderer; o Isaac Sim só é necessário para abrir o USDA terminado.

### Reproduzir o ambiente da Urgência

O perfil `configs/urgencia.json` contém os parâmetros usados no hospital,
incluindo o recorte que remove o exterior e o teto dos corredores:

```bash
python scripts/run_pipeline.py \
  "./data/urgencia.ply" \
  "./output_environment_urgencia" \
  --profile configs/urgencia.json
```

Este perfil só deve ser aplicado ao mesmo PLY ou a uma exportação com o mesmo
referencial. Os limites locais do recorte são:

| Eixo | Mínimo | Máximo |
|---|---:|---:|
| X | -55,00 m | 55,00 m |
| Y | 12,00 m | 70,00 m |
| Z | 2,15 m | 4,35 m |

O corte superior em Z=4,35 m remove o teto. No resultado validado original, o
chão estava aproximadamente em Z=2,3175 m.

## Vários ficheiros PLY

Só se devem juntar PLYs que já estejam no mesmo sistema de coordenadas e
tenham unidades e orientação compatíveis:

```bash
python scripts/merge_point_clouds.py \
  "./data/tile_01.ply" \
  "./data/tile_02.ply" \
  "./data/merged.ply" \
  --metadata-output "./data/merged_metadata.json"
```

Se os scans não estiverem registados no mesmo referencial, é necessário fazer
primeiro o registo/alinhamento; concatená-los produziria geometria duplicada.

## Como os parâmetros automáticos são escolhidos

O pipeline mede a distância mediana ao vizinho mais próximo numa amostra da
nuvem original. Quando o respetivo valor no perfil é zero:

- voxel de limpeza: 6 × espaçamento mediano;
- voxel visual: máximo entre o voxel de limpeza e 10 × o espaçamento;
- voxel de colisão: máximo entre 8 × o voxel visual e 80 × o espaçamento;
- alternativa em pontos: máximo entre 2 × o voxel visual e 20 × o espaçamento;
- profundidade Poisson visual: 11 ou 12, conforme o número de pontos;
- profundidade Poisson da colisão: 8 ou 9.

Todos os valores efetivamente selecionados ficam em
`reports/processing_report.json`.

No caso validado da Urgência foram usados:

- limpeza: voxel 0,03 m, 16 vizinhos e `std_ratio=4,0`;
- visual: voxel 0,05 m, Poisson depth 12, corte de densidade 0,5%;
- colisão: voxel 0,40 m, Poisson depth 9 e alvo de 80 000 triângulos;
- pontos: voxel 0,10 m e largura 0,09 m.

## Pipeline

1. Lê o cabeçalho e a geometria do PLY.
2. Verifica XYZ, RGB, valores inválidos, dimensões e densidade.
3. Recentra coordenadas grandes para melhorar a precisão numérica.
4. Remove NaN, infinitos, duplicados por voxel e outliers conservadores.
5. Estima e orienta normais.
6. Reconstrói a mesh visual com Poisson.
7. Remove apenas baixa densidade e regiões sem suporte próximo.
8. Transfere RGB dos pontos para os vértices.
9. Constrói uma colisão separada a partir de uma nuvem mais reduzida.
10. Escreve diretamente os USD com `primvars:displayColor`.
11. Monta e valida `environment_ready.usda`.

Poisson é o método principal por ser geralmente mais estável em scans
ambientais densos e ruidosos. Para estruturas finas ou superfícies muito
abertas, a representação em pontos pode ser visualmente mais fiel.

## Porque é que a mesh mantém as cores

As cores não dependem de uma textura externa. São guardadas em cada vértice
como `primvars:displayColor` e ligadas explicitamente a
`UsdPreviewSurface.diffuseColor` através de
`UsdPrimvarReader_float3`.

Isto evita o problema em que alguns conversores GLB/OBJ criam um material
branco que ignora as cores dos vértices.

## Estrutura criada

```text
output_environment/
├── environment_ready.usda
├── environment_points_ready.usda
├── visual/
│   ├── visual_mesh.usd
│   ├── visual_mesh.ply
│   ├── visual_mesh.obj
│   ├── visual_mesh.glb
│   └── visual_points.usd
├── collision/
│   ├── collision_mesh.usd
│   ├── collision_mesh.ply
│   ├── collision_mesh.obj
│   └── collision_mesh.glb
├── pointcloud/
│   └── map_clean.ply
├── scripts/
├── reports/
└── README.md
```

Os ficheiros NPZ intermédios também são mantidos no resultado para permitir
recriar os USD sem repetir a reconstrução Poisson.

## Organização no Isaac Sim

```text
/World
└── Environment
    ├── Visual
    ├── PointsAlternative
    ├── Collision
    └── GroundCollider
```

- `Visual`: visível e sem colisão;
- `PointsAlternative`: oculto na stage principal;
- `Collision`: invisível e com collider ativo;
- `GroundCollider`: invisível e colocado sob a geometria.

## Transportar para outro computador

Copiar a pasta de saída completa. Não separar o USDA das subpastas
`visual` e `collision`. Como as referências são relativas, a pasta pode
ser colocada em qualquer localização no computador de destino.

O PLY original e este repositório não são necessários para simplesmente abrir
o ambiente já terminado.

## Validação

O passo final confirma automaticamente:

- existência de todos os ficheiros referenciados;
- ausência de caminhos absolutos no USDA;
- `metersPerUnit = 1` e Z-up;
- hierarquia esperada;
- cores e opacidade da mesh visual;
- colisão desativada na visual;
- collider ativo e invisível na mesh de colisão;
- bounding box final;
- cores na alternativa em pontos.

Resultados:

- `reports/validation_report.json`;
- `reports/validation_points_report.json`;
- `reports/report.md`;
- `reports/previews/`.

## Dados grandes

Os PLY, meshes, NPZ e USD gerados podem ocupar vários gigabytes e estão
excluídos pelo `.gitignore`. O GitHub guarda o código e os parâmetros; os
dados devem ser conservados localmente, num disco externo ou num sistema de
armazenamento próprio.

## Testes do repositório

```bash
python -m unittest discover -s tests -v
```

Os testes verificam sintaxe, perfis JSON e ausência de caminhos específicos da
máquina onde o projeto foi criado.

## Licença

MIT. Consultar `LICENSE`.
