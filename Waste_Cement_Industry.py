import time
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from pyomo.environ import (
    ConcreteModel, Var, NonNegativeReals, Objective, Constraint,
    SolverFactory, value, minimize
)
from pyomo.opt import SolverStatus, TerminationCondition

# ============================================================
# CONSTANTES
# ============================================================
BAG_WEIGHT_KG = 0.050
ENERGY_INTENSITY_CP_GJ_PER_T = 3500
TRUCK_PAYLOAD_T = 20.0
LHV_RESIDUE_MJ_PER_KG = 10.0
DIESEL_CONSUMPTION_L_PER_KM = 0.30
EF_DIESEL_KGCO2_PER_L = 2.67

GJ_PER_TEP = 41.868
EF_TRAD_KGCO2_PER_GJ = 92.8
EF_RESID_FRACTION = 0.05  # Fator de emissão de resíduo relativo ao tradicional

# ============================================================
# PARÂMETROS ECONÔMICOS - DIFERENCIADOS POR TIPO
# ============================================================
CAMBIO = 5.5

# --- RESÍDUO URBANO (MSW) ---
C_MSW_R_PER_GJ = 0.0 / GJ_PER_TEP  # Custo base do resíduo MSW (R$/GJ)
CAPEX_MSW_USD_PER_T = 30.0         # Investimento em infraestrutura MSW
OPEX_MSW_USD_PER_T = 2.5           # Custo operacional MSW

# --- RESÍDUO AGRÍCOLA (AGRO) ---
C_AGRO_R_PER_GJ = 0.0 / GJ_PER_TEP  # Custo base do resíduo AGRO (R$/GJ)
CAPEX_AGRO_USD_PER_T = 30.0         # Investimento em infraestrutura AGRO
OPEX_AGRO_USD_PER_T = 2.5           # Custo operacional AGRO

# --- RESÍDUO INDUSTRIAL (IW) ---
C_IW_R_PER_GJ = 0.0 / GJ_PER_TEP     # Custo base do resíduo IW (R$/GJ)
CAPEX_IW_USD_PER_T = 30.0            # Investimento em infraestrutura IW
OPEX_IW_USD_PER_T = 2.5              # Custo operacional IW

# --- COMBUSTÍVEL TRADICIONAL ---
C_TRAD_R_PER_GJ = 600.0 / GJ_PER_TEP  # Custo combustível tradicional

# --- TRANSPORTE ---
COST_USD_PER_GJ_PER_KM = 0.01      # Custo de transporte (USD/GJ/km)

# Parâmetros de otimização
ALPHA = 0.3  # Taxa de substituição (50%)
MAX_DIST_KM = 300.0

#%%
# ============================================================
# 0) PATH
# ============================================================
path = r"C:\Users\ottoh\OneDrive\Meus artigos\Resíduos na produção de cimento"

#%%
# ============================================================
# FUNÇÃO AUXILIAR: Calcular custo base por tipo
# ============================================================
def calcular_custo_base_tipo():
    """
    Calcula o custo base (sem transporte) para cada tipo de resíduo.
    Retorna dict: {'MSW': ..., 'AGRO': ..., 'IW': ...}
    """
    # Observação: numericamente, MJ/kg == GJ/t
    GJ_PER_TON_RESID = LHV_RESIDUE_MJ_PER_KG  # ex.: 10 MJ/kg => 10 GJ/t

    def _custo_base(C_R_per_GJ, CAPEX_USD_per_T, OPEX_USD_per_T):
        capex_R_per_GJ = (CAPEX_USD_per_T / GJ_PER_TON_RESID) * CAMBIO
        opex_R_per_GJ  = (OPEX_USD_per_T  / GJ_PER_TON_RESID) * CAMBIO
        return float(C_R_per_GJ + capex_R_per_GJ + opex_R_per_GJ)

    return {
        "MSW": _custo_base(C_MSW_R_PER_GJ,  CAPEX_MSW_USD_PER_T,  OPEX_MSW_USD_PER_T),
        "AGRO": _custo_base(C_AGRO_R_PER_GJ, CAPEX_AGRO_USD_PER_T, OPEX_AGRO_USD_PER_T),
        "IW": _custo_base(C_IW_R_PER_GJ,    CAPEX_IW_USD_PER_T,   OPEX_IW_USD_PER_T),
    }

# Calcular custos base
C_base_tipo = calcular_custo_base_tipo()
C_transp_factor = COST_USD_PER_GJ_PER_KM * CAMBIO  # R$/GJ/km

print("\n=== PARÂMETROS ECONÔMICOS DIFERENCIADOS ===")
print(f"Alpha (substituição): {ALPHA:.0%}")
print(f"Distância máxima: {MAX_DIST_KM} km")
print(f"\nCombustível Tradicional:")
print(f"  Custo: {C_TRAD_R_PER_GJ:.4f} R$/GJ")
print(f"\nResíduo Urbano (MSW):")
print(f"  Custo matéria-prima: {C_MSW_R_PER_GJ:.4f} R$/GJ")
print(f"  CAPEX: {(CAPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  OPEX: {(OPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  → TOTAL (sem transporte): {C_base_tipo['MSW']:.2f} R$/GJ")
print(f"\nResíduo Agrícola (AGRO):")
print(f"  Custo matéria-prima: {C_AGRO_R_PER_GJ:.4f} R$/GJ")
print(f"  CAPEX: {(CAPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  OPEX: {(OPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  → TOTAL (sem transporte): {C_base_tipo['AGRO']:.2f} R$/GJ")
print(f"\nResíduo Industrial (IW):")
print(f"  Custo matéria-prima: {C_IW_R_PER_GJ:.4f} R$/GJ")
print(f"  CAPEX: {(CAPEX_IW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  OPEX: {(OPEX_IW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ")
print(f"  → TOTAL (sem transporte): {C_base_tipo['IW']:.2f} R$/GJ")
print(f"\nTransporte:")
print(f"  Custo: {C_transp_factor:.4f} R$/GJ/km")

#%%
# ============================================================
# 1) Ler shapefile dos municípios
# ============================================================
print("\n>>> Carregando shapefile de municípios...")
gdf_mun = gpd.read_file(path + "/BR_Municipios_2024.shp")

# Validações
if gdf_mun.empty:
    raise ValueError("Shapefile de municípios vazio ou não encontrado")
    
if not {'CD_MUN', 'geometry'}.issubset(gdf_mun.columns):
    raise ValueError("Shapefile deve conter colunas 'CD_MUN' e 'geometry'")

gdf_mun["CD_MUN"] = gdf_mun["CD_MUN"].astype(str).str.zfill(7)
print(f"✓ {len(gdf_mun):,} municípios carregados")

#%%
# ============================================================
# 2) Ler potenciais energéticos (MSW + Agro) - MANTER SEPARADO
# ============================================================
print("\n>>> Carregando dados de potencial energético...")
df_msw = pd.read_csv(path + "/MSW_Energy_potential.csv")
df_agro = pd.read_csv(path + "/AgroWaste_Energy_potential.csv")
df_iw = pd.read_csv(path + "/IW_Energy_potential.csv")


df_msw["CD_MUN"] = df_msw["CD_MUN"].astype(str).str.zfill(7)
df_msw['CD_MUN'] = df_msw['CD_MUN'].str[1:]

df_agro["CD_MUN"] = df_agro["CD_MUN"].astype(str).str.zfill(7)
df_agro["CD_MUN_6"] = df_agro["CD_MUN"].str[:-1]

df_iw["CD_MUN"] = df_iw["CD_MUN"].astype(str).str.zfill(7)


mapa_cd_mun = (
    df_agro[["CD_MUN_6", "CD_MUN"]]
    .drop_duplicates()
    .set_index("CD_MUN_6")["CD_MUN"]
)

df_msw["CD_MUN"] = df_msw["CD_MUN"].map(mapa_cd_mun)
# df_iw["CD_MUN"] = df_iw["CD_MUN"].map(mapa_cd_mun)


df = df_msw.merge(df_agro, on="CD_MUN", how="outer")
df = df.merge(df_iw, on="CD_MUN", how="outer")

df["MSW_GJ"] = df["Energy potential (GJ)"].fillna(0.0)
df["AGRO_GJ"] = df["Total (GJ)"].fillna(0.0)
df["IW_GJ"] = df["IW_GJ"].fillna(0.0)

df["potencial_total_GJ"] = df["MSW_GJ"] + df["AGRO_GJ"] + df["IW_GJ"]

df_final = df[["CD_MUN", "MSW_GJ", "AGRO_GJ", "IW_GJ", "potencial_total_GJ"]].copy()
gdf_merge = gdf_mun.merge(df_final, on="CD_MUN", how="left")

print(f"✓ Potencial energético carregado: {df_final['potencial_total_GJ'].sum():,.0f} GJ total")
print(f"  - MSW:  {df_final['MSW_GJ'].sum():,.0f} GJ")
print(f"  - AGRO: {df_final['AGRO_GJ'].sum():,.0f} GJ")
print(f"  - IW:   {df_final['IW_GJ'].sum():,.0f} GJ")

#%%
# ============================================================
# 3) Ler fábricas (pontos)
# ============================================================
print("\n>>> Carregando dados de fábricas...")
df_fab = pd.read_csv(path + "/Fabricas_geo.csv")

if df_fab.empty:
    raise ValueError("Arquivo de fábricas vazio")

df_fab["Name"] = df_fab["Name"].astype(str).str.strip()

# Validar georreferenciamento
invalid_geo = df_fab[~df_fab['Georreferenciamento'].str.contains(',', na=False)]
if not invalid_geo.empty:
    print(f"⚠️ {len(invalid_geo)} fábricas com georreferenciamento inválido (serão removidas)")
    df_fab = df_fab[df_fab['Georreferenciamento'].str.contains(',', na=False)]

df_fab[["lat", "lon"]] = df_fab["Georreferenciamento"].str.split(",", expand=True)
df_fab["lat"] = pd.to_numeric(df_fab["lat"], errors='coerce')
df_fab["lon"] = pd.to_numeric(df_fab["lon"], errors='coerce')

# Remover coordenadas inválidas
if df_fab[['lat', 'lon']].isna().any().any():
    print("⚠️ Algumas fábricas têm coordenadas inválidas (serão removidas)")
    df_fab = df_fab.dropna(subset=['lat', 'lon'])

gdf_fab = gpd.GeoDataFrame(
    df_fab,
    geometry=gpd.points_from_xy(df_fab["lon"], df_fab["lat"]),
    crs="EPSG:4326"
)

# Alinhar CRS do mapa
gdf_fab = gdf_fab.to_crs(gdf_mun.crs)
print(f"✓ {len(gdf_fab):,} fábricas carregadas")

#%%
# ============================================================
# 4) Plot rápido (opcional)
# ============================================================
# print("\n>>> Gerando mapa de potencial energético...")
# fig, ax = plt.subplots(figsize=(12, 10))
# gdf_merge.plot(column="potencial_total_GJ", cmap="viridis", legend=True,
#                 linewidth=0.1, edgecolor="black", ax=ax)
# gdf_fab.plot(ax=ax, color="red", markersize=40, label="Plantas de cimento")
# plt.title("Potencial Energético (MSW+Agro) + Localização das Plantas")
# plt.legend()
# plt.axis("off")
# plt.tight_layout()
# plt.savefig(path + "/mapa_potencial_plantas.png", dpi=150, bbox_inches='tight')
# plt.show()
# print("✓ Mapa salvo: mapa_potencial_plantas.png")

#%%
# ============================================================
# 5) Construir demanda por planta
# ============================================================
print("\n>>> Calculando demanda por planta...")
DF_plant_waste_by_city = pd.read_csv(path + '/Intersection_radius_cement_plants_R03_v3.csv')
DF_plant_waste_by_city["Name"] = DF_plant_waste_by_city["Name"].astype(str).str.strip()

# Limpeza de dados - versão melhorada
DF_plant_waste_by_city['Cap__Mensal__scs_'] = (
    DF_plant_waste_by_city['Cap__Mensal__scs_']
    .astype(str)
    .str.replace(r'[,.\n]|sacos', '', regex=True)
    .astype('int64')
)
DF_plant_waste_by_city['Capacidade_ins_(t)'] = (
    DF_plant_waste_by_city['Cap__Mensal__scs_'] * BAG_WEIGHT_KG * 12
)

# Extrair UF - versão otimizada
DF_plant_waste_by_city['UF'] = DF_plant_waste_by_city['Cidade'].str[-2:]

# Correções manuais
DF_plant_waste_by_city.loc[
    DF_plant_waste_by_city.Name == 'Cimento Uau - Pains/MG', 'UF'
] = 'MG'
DF_plant_waste_by_city.loc[
    DF_plant_waste_by_city.Name == "Cimento Forte - Suape/PE", 'UF'
] = 'PE'

# Capacidade por planta - versão otimizada
Capacidade_plantas = pd.DataFrame(
    index=DF_plant_waste_by_city.Name.unique(),
    columns=['Capacidade anual', 'UF'],
    data=0.0
)

# Usar mapeamento ao invés de loop
cap_dict = (
    DF_plant_waste_by_city
    .drop_duplicates('Name')
    .set_index('Name')['Capacidade_ins_(t)']
    .to_dict()
)
uf_dict = (
    DF_plant_waste_by_city
    .drop_duplicates('Name')
    .set_index('Name')['UF']
    .to_dict()
)

Capacidade_plantas['Capacidade anual'] = Capacidade_plantas.index.map(cap_dict)
Capacidade_plantas['UF'] = Capacidade_plantas.index.map(uf_dict)

# Produção por UF
Producao_UF_2019 = pd.read_csv(path + '/producao_cimento_UF_2019.csv', sep=',')
Producao_clinquer_2019 = 63000 * 0.71  # kt

# Produção por planta
Producao_plantas = pd.DataFrame(
    index=Capacidade_plantas.index,
    columns=['Producao (kt)', 'Tipo', 'Producao clinquer (kt)'],
    data=0.0
)

# Capacidade total por UF
cap_total_uf = Capacidade_plantas.groupby('UF')['Capacidade anual'].sum()

for planta in Producao_plantas.index:
    uf_planta = Capacidade_plantas.loc[planta, 'UF']
    cap_planta = float(Capacidade_plantas.loc[planta, 'Capacidade anual'])
    cap_total = float(cap_total_uf.loc[uf_planta])
    
    prod_uf = Producao_UF_2019.loc[
        Producao_UF_2019['UF'] == uf_planta, 'Producao (kt) Ajustada'
    ]
    
    if not prod_uf.empty:
        Producao_plantas.loc[planta, 'Producao (kt)'] = (
            (cap_planta / cap_total) * float(prod_uf.iloc[0])
        )
    
    tipo = DF_plant_waste_by_city.loc[
        DF_plant_waste_by_city['Name'] == planta, 'Tipo_Planta'
    ]
    if not tipo.empty:
        Producao_plantas.loc[planta, 'Tipo'] = tipo.unique()[0]

# Produção de clínquer
total_prod_fabricas = Producao_plantas.loc[
    Producao_plantas['Tipo'] == 'Fábrica', 'Producao (kt)'
].sum()

for planta in Producao_plantas.index:
    if Producao_plantas.loc[planta, 'Tipo'] == 'Fábrica':
        Producao_plantas.loc[planta, 'Producao clinquer (kt)'] = (
            Producao_plantas.loc[planta, 'Producao (kt)'] / total_prod_fabricas
        ) * Producao_clinquer_2019
    else:
        Producao_plantas.loc[planta, 'Producao clinquer (kt)'] = 0.0

# Demanda energética
Energy_demand_plant = (
    Producao_plantas['Producao clinquer (kt)'] * ENERGY_INTENSITY_CP_GJ_PER_T
).to_frame()
Energy_demand_plant = Energy_demand_plant.rename(
    columns={'Producao clinquer (kt)': 'Demand_GJ'}
)
Energy_demand_plant["Name"] = Energy_demand_plant.index.astype(str).str.strip()

df_demand = Energy_demand_plant[["Name", "Demand_GJ"]].copy()
df_demand = df_demand.drop_duplicates(subset="Name", keep="first").reset_index(drop=True)

# Merge demanda em gdf_fab
gdf_fab = gdf_fab.copy()
gdf_fab["Name"] = gdf_fab["Name"].astype(str).str.strip()
if "Demand_GJ" in gdf_fab.columns:
    gdf_fab = gdf_fab.drop(columns=["Demand_GJ"])
gdf_fab = gdf_fab.merge(df_demand, on="Name", how="left")
gdf_fab = gdf_fab[gdf_fab["Demand_GJ"].fillna(0) > 0].copy()

print(f"✓ Demanda calculada para {len(gdf_fab)} plantas")
print(f"  Demanda total: {gdf_fab['Demand_GJ'].sum():,.0f} GJ")

#%%
# ============================================================
# 6) Parâmetros de emissões e transporte
# ============================================================
# Transporte / emissões
GJ_per_truck = (TRUCK_PAYLOAD_T * 1000 * LHV_RESIDUE_MJ_PER_KG) / 1000.0
EF_resid_kgCO2_per_GJ = EF_RESID_FRACTION * EF_TRAD_KGCO2_PER_GJ

#%%
# ============================================================
# 7) Preparar dados para otimização (CRS métrico)
# ============================================================
print("\n>>> Preparando dados para otimização...")
gdf_fab_m = gdf_fab.to_crs(5880).copy()
gdf_mun_m = gdf_merge.to_crs(5880).copy()

gdf_fab_m["Name"] = gdf_fab_m["Name"].astype(str).str.strip()
gdf_mun_m["CD_MUN"] = gdf_mun_m["CD_MUN"].astype(str).str.zfill(7)

gdf_mun_m["MSW_GJ"] = gdf_mun_m["MSW_GJ"].fillna(0.0)
gdf_mun_m["AGRO_GJ"] = gdf_mun_m["AGRO_GJ"].fillna(0.0)
gdf_mun_m["IW_GJ"] = gdf_mun_m["IW_GJ"].fillna(0.0)

gdf_mun_m = gdf_mun_m[(gdf_mun_m["MSW_GJ"] + gdf_mun_m["AGRO_GJ"] + gdf_mun_m["IW_GJ"]) > 0].copy()


# Centroides para distância
gdf_mun_m["centroid"] = gdf_mun_m.geometry.centroid

# Conjuntos
plants = gdf_fab_m[["Name", "Demand_GJ", "geometry"]].copy()
mun = gdf_mun_m[["CD_MUN", "MSW_GJ", "AGRO_GJ", "IW_GJ", "centroid"]].copy()

Demand = plants.set_index("Name")["Demand_GJ"].to_dict()

Supply_type = {}
for _, r in mun.iterrows():
    m = r["CD_MUN"]
    Supply_type[(m, "MSW")] = float(r["MSW_GJ"])
    Supply_type[(m, "AGRO")] = float(r["AGRO_GJ"])
    Supply_type[(m, "IW")]  = float(r["IW_GJ"])


P = list(Demand.keys())
M = sorted({m for (m, t) in Supply_type.keys()})
T = ["MSW", "AGRO", "IW"]

plant_geom = plants.set_index("Name")["geometry"]
mun_cent = mun.set_index("CD_MUN")["centroid"]

print(f"✓ {len(P)} plantas")
print(f"✓ {len(M)} municípios com oferta")
print(f"✓ 2 tipos de resíduo (MSW, AGRO)")

#%%
# ============================================================
# 8) Função auxiliar: calcular distâncias viáveis
# ============================================================
def calcular_distancias_viaveis(plantas_gdf, municipios_gdf, max_dist_km):
    """
    Calcula distâncias entre plantas e municípios <= max_dist_km.
    
    Retorna dict {(planta, municipio): distancia_km}
    """
    dist_dict = {}
    plant_geom_idx = plantas_gdf.set_index("Name")["geometry"]
    mun_cent_idx = municipios_gdf.set_index("CD_MUN")["centroid"]
    
    for p in plant_geom_idx.index:
        distancias_km = mun_cent_idx.distance(plant_geom_idx.loc[p]) / 1000.0
        distancias_viaveis = distancias_km[distancias_km <= max_dist_km]
        for m, d in distancias_viaveis.items():
            dist_dict[(p, m)] = float(d)
    
    return dist_dict

#%%
# ============================================================
# 9) Distâncias <= MAX_DIST_KM e pares (p,m,t)
# ============================================================
t0 = time.time()
print(f"\n>>> Calculando distâncias (máx {MAX_DIST_KM} km)...")

dist_km = calcular_distancias_viaveis(plants, mun, MAX_DIST_KM)

# Pares com tipo só onde há oferta
PMT = []
for (p, m), d in dist_km.items():
    for t in ["MSW", "AGRO", "IW"]:
        if Supply_type.get((m, t), 0.0) > 0:
            PMT.append((p, m, t))

# ============================================================
# CUSTO POR PAR - DIFERENCIADO POR TIPO
# ============================================================
c_pmt = {}
for (p, m, t) in PMT:
    # Custo base depende do tipo de resíduo
    custo_base = C_base_tipo[t]
    # Custo de transporte é igual para ambos
    custo_transporte = C_transp_factor * dist_km[(p, m)]
    # Custo total
    c_pmt[(p, m, t)] = custo_base + custo_transporte

# Municípios ativos por tipo
M_active_type = sorted({(m, t) for (_, m, t) in PMT})

print(f"✓ Pares (planta, município) viáveis: {len(dist_km):,}")
print(f"✓ Variáveis de decisão (p,m,t): {len(PMT):,}")
print(f"✓ Tempo: {time.time()-t0:.1f}s")

# Checagem de viabilidade
total_demand = sum(Demand.values())
target = ALPHA * total_demand
accessible_supply = sum(Supply_type[(m, t)] for (m, t) in M_active_type)

print(f"\n=== ANÁLISE DE VIABILIDADE ===")
print(f"Demanda total: {total_demand:,.0f} GJ")
print(f"Meta substituição ({ALPHA:.0%}): {target:,.0f} GJ")
print(f"Oferta acessível (<= {MAX_DIST_KM} km): {accessible_supply:,.0f} GJ")

if accessible_supply + 1e-6 < target:
    print("⚠️ ATENÇÃO: Provável inviabilidade!")
    print("   Oferta acessível < meta de substituição")
    print("   Sugestões: aumentar MAX_DIST_KM ou reduzir ALPHA")

# Validação final
if not PMT:
    raise ValueError(
        "Nenhum par (planta, município, tipo) viável encontrado. "
        "Aumente MAX_DIST_KM ou verifique os dados de entrada."
    )

#%%
# ============================================================
# 10) Pyomo - minimizar custo, substituição TOTAL
# ============================================================
print("\n>>> Construindo modelo Pyomo...")
t0 = time.time()

model = ConcreteModel()
model.x = Var(PMT, domain=NonNegativeReals)

def obj_rule(mod):
    return sum(mod.x[p, m, t] * c_pmt[(p, m, t)] for (p, m, t) in PMT)

model.obj = Objective(rule=obj_rule, sense=minimize)

# (i) Substituição TOTAL
model.total_sub = Constraint(
    expr=sum(model.x[p, m, t] for (p, m, t) in PMT) == ALPHA * total_demand
)

# (ii) Limite por planta
def plant_cap_rule(mod, p):
    return sum(mod.x[p, m, t] for (pp, m, t) in PMT if pp == p) <= Demand[p]

model.plant_cap = Constraint(P, rule=plant_cap_rule)

# (iii) Oferta por município e tipo
def supply_type_rule(mod, m, t):
    return sum(
        mod.x[p, m, t] for (p, mm, tt) in PMT if mm == m and tt == t
    ) <= Supply_type[(m, t)]

model.supply_type = Constraint(M_active_type, rule=supply_type_rule)

print(f"✓ Modelo construído em {time.time()-t0:.1f}s")
print(f"  Variáveis de decisão: {len(PMT):,}")
print(f"  Restrições: {len(M_active_type) + len(P) + 1:,}")
print(f"  Memória estimada: ~{len(PMT) * 8 / 1e6:.1f} MB")

#%%
# ============================================================
# 11) Resolver com solver
# ============================================================
print("\n>>> Resolvendo otimização...")
t0_solve = time.time()

solver_name = "ipopt"
try:
    from amplpy import modules
    solver = SolverFactory(
        solver_name + "nl",
        executable=modules.find(solver_name),
        solve_io="nl"
    )
    
    result = solver.solve(model, tee=True)
    
    # Verificar status
    if result.solver.status != SolverStatus.ok:
        raise RuntimeError(f"Solver falhou: {result.solver.status}")
    
    if result.solver.termination_condition != TerminationCondition.optimal:
        print(f"⚠️ Solução pode não ser ótima: {result.solver.termination_condition}")
    else:
        print("✓ Solução ótima encontrada")
    
except Exception as e:
    print(f"❌ Erro na otimização: {e}")
    raise

elapsed_solve = time.time() - t0_solve
print(f"✓ Tempo de solução: {elapsed_solve:.1f}s")

#%%
# ============================================================
# 12) EXTRAIR RESULTADOS + % MSW/AGRO + emissões/custos/CMA
# ============================================================
print("\n>>> Extraindo resultados...")
t0 = time.time()

rows = []
for (p, m, t) in PMT:
    xval = value(model.x[p, m, t])
    if xval is not None and xval > 1e-6:
        gj = float(xval)
        dkm = float(dist_km[(p, m)])
        cunit = float(c_pmt[(p, m, t)])
        rows.append({
            "Name": p,
            "CD_MUN": m,
            "Tipo": t,
            "GJ_alocado": gj,
            "dist_km": dkm,
            "custo_unit_R_per_GJ": cunit,
            "custo_base_R_per_GJ": C_base_tipo[t],  # Adiciona custo base específico
            "custo_transporte_R_per_GJ": C_transp_factor * dkm,  # Custo transporte separado
            "custo_total_R": gj * cunit
        })

df_flow = pd.DataFrame(rows)

if df_flow.empty:
    print("⚠️ df_flow vazio: modelo pode estar inviável ou alpha=0.")
    # Criar arquivo de erro
    out_xlsx = path + f"/otimizacao_ERRO_alpha_{int(ALPHA*100)}pct.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([{
            "msg": "df_flow vazio - verifique viabilidade (alpha, MAX_DIST, oferta)",
            "solver_status": str(result.solver.status),
            "termination_condition": str(result.solver.termination_condition)
        }]).to_excel(writer, sheet_name="erro", index=False)
    print(f"Arquivo de erro criado: {out_xlsx}")
else:
    # Transporte e emissões
    df_flow["dist_roundtrip_km"] = 2.0 * df_flow["dist_km"]
    df_flow["n_trips"] = np.ceil(df_flow["GJ_alocado"] / GJ_per_truck)
    df_flow["total_dist_km"] = df_flow["n_trips"] * df_flow["dist_roundtrip_km"]
    df_flow["litros_diesel"] = df_flow["total_dist_km"] * DIESEL_CONSUMPTION_L_PER_KM
    df_flow["E_transporte_kgCO2"] = df_flow["litros_diesel"] * EF_DIESEL_KGCO2_PER_L
    
    df_flow["E_original_kgCO2"] = df_flow["GJ_alocado"] * EF_TRAD_KGCO2_PER_GJ
    df_flow["E_nova_comb_kgCO2"] = df_flow["GJ_alocado"] * EF_resid_kgCO2_per_GJ
    df_flow["E_nova_total_kgCO2"] = (
        df_flow["E_nova_comb_kgCO2"] + df_flow["E_transporte_kgCO2"]
    )
    
    df_flow["Delta_emissoes_kgCO2"] = (
        df_flow["E_original_kgCO2"] - df_flow["E_nova_total_kgCO2"]
    )
    
    df_flow["Custo_original_R"] = df_flow["GJ_alocado"] * C_TRAD_R_PER_GJ
    df_flow["Delta_custo_R"] = df_flow["custo_total_R"] - df_flow["Custo_original_R"]
    
    # CMA usando pd.NA
    df_flow["CMA_R_per_tCO2"] = df_flow["Delta_custo_R"].where(
        df_flow["Delta_emissoes_kgCO2"] > 0,
        pd.NA
    ) * 1000.0 / df_flow["Delta_emissoes_kgCO2"]
    
    # Demanda original por planta
    df_demand_plants = pd.DataFrame({
        "Name": list(Demand.keys()),
        "Demand_original_GJ": list(Demand.values())
    })
    
    # Agregação por planta
    df_plant = (
        df_flow.groupby("Name", as_index=False)
        .agg(
            Demand_substituida_GJ=("GJ_alocado", "sum"),
            custo_novo_R=("custo_total_R", "sum"),
            custo_original_R=("Custo_original_R", "sum"),
            Delta_custo_R=("Delta_custo_R", "sum"),
            Emissao_original_kgCO2=("E_original_kgCO2", "sum"),
            Emissao_nova_kgCO2=("E_nova_total_kgCO2", "sum"),
            Delta_emissoes_kgCO2=("Delta_emissoes_kgCO2", "sum"),
            dist_media_km=(
                "dist_km",
                lambda s: np.average(
                    s, weights=df_flow.loc[s.index, "GJ_alocado"]
                )
            ),
            litros_diesel=("litros_diesel", "sum"),
            total_dist_km=("total_dist_km", "sum"),
        )
    )
    
    df_plant = df_plant.merge(df_demand_plants, on="Name", how="left")
    df_plant["Perc_substituido_%"] = (
        100.0 * df_plant["Demand_substituida_GJ"] / df_plant["Demand_original_GJ"]
    )
    
    df_plant["CMA_R_per_tCO2"] = df_plant["Delta_custo_R"].where(
        df_plant["Delta_emissoes_kgCO2"] > 0,
        pd.NA
    ) * 1000.0 / df_plant["Delta_emissoes_kgCO2"]
    
    # % MSW vs % AGRO (sistema)
    total_gj = df_flow["GJ_alocado"].sum()
    by_type = df_flow.groupby("Tipo")["GJ_alocado"].sum()
    
    perc_msw = 100.0 * by_type.get("MSW", 0.0) / total_gj if total_gj > 0 else 0.0
    perc_agro = 100.0 * by_type.get("AGRO", 0.0) / total_gj if total_gj > 0 else 0.0
    perc_iw = 100.0 * by_type.get("IW", 0.0) / total_gj if total_gj > 0 else 0.
    
    # Custos por tipo
    by_type_custo = df_flow.groupby("Tipo")["custo_total_R"].sum()
    custo_msw_total = by_type_custo.get("MSW", 0.0)
    custo_agro_total = by_type_custo.get("AGRO", 0.0)
    custo_iw_total = by_type_custo.get("IW", 0.0)
    
    # Resumo do sistema
    total_cost_new = df_flow["custo_total_R"].sum()
    total_cost_old = df_flow["Custo_original_R"].sum()
    total_delta_cost = df_flow["Delta_custo_R"].sum()
    
    total_E_old = df_flow["E_original_kgCO2"].sum()
    total_E_new = df_flow["E_nova_total_kgCO2"].sum()
    total_delta_E = df_flow["Delta_emissoes_kgCO2"].sum()
    
    CMA_system = (
        (total_delta_cost * 1000.0 / total_delta_E)
        if total_delta_E > 0
        else np.nan
    )
    
    df_summary = pd.DataFrame([{
        "alpha": ALPHA,
        "MAX_DIST_KM": MAX_DIST_KM,
        "GJ_substituidos_total": total_gj,
        "Perc_MSW_%": perc_msw,
        "Perc_AGRO_%": perc_agro,
        "Perc_IW_%": perc_iw,
        "GJ_MSW": by_type.get("MSW", 0.0),
        "GJ_AGRO": by_type.get("AGRO", 0.0),
        "GJ_IW": by_type.get("IW", 0.0),
        "Custo_MSW_R": custo_msw_total,
        "Custo_AGRO_R": custo_agro_total,
        "Custo_IW_R": custo_iw_total,
        "Custo_base_MSW_R_per_GJ": C_base_tipo['MSW'],
        "Custo_base_AGRO_R_per_GJ": C_base_tipo['AGRO'],
        "Custo_base_IW_R_per_GJ": C_base_tipo['IW'],
        "custo_novo_total_R": total_cost_new,
        "custo_original_total_R": total_cost_old,
        "Delta_custo_total_R": total_delta_cost,
        "Emissao_original_total_kgCO2": total_E_old,
        "Emissao_nova_total_kgCO2": total_E_new,
        "Delta_emissoes_total_kgCO2": total_delta_E,
        "CMA_sistema_R_per_tCO2": CMA_system,
        "tempo_solucao_s": elapsed_solve,
        "solver_status": str(result.solver.status),
        "termination_condition": str(result.solver.termination_condition)
    }])
    
    print("\n" + "="*60)
    print("RESULTADO DO SISTEMA")
    print("="*60)
    print(f"Substituição alvo (alpha): {ALPHA:.1%}")
    print(f"GJ substituídos: {total_gj:,.0f} GJ")
    print(f"  - MSW:  {by_type.get('MSW', 0.0):,.0f} GJ ({perc_msw:.1f}%)")
    print(f"  - AGRO: {by_type.get('AGRO', 0.0):,.0f} GJ ({perc_agro:.1f}%)")
    print(f"  - IW: {by_type.get('IW', 0.0):,.0f} GJ ({perc_iw:.1f}%)")
    print(f"\nCustos por tipo:")
    print(f"  MSW:  {custo_msw_total:,.0f} R$ (base: {C_base_tipo['MSW']:.2f} R$/GJ)")
    print(f"  AGRO: {custo_agro_total:,.0f} R$ (base: {C_base_tipo['AGRO']:.2f} R$/GJ)")
    print(f"  IW: {custo_agro_total:,.0f} R$ (base: {C_base_tipo['IW']:.2f} R$/GJ)")
    print(f"\nCustos totais:")
    print(f"  Custo novo:      {total_cost_new:,.0f} R$")
    print(f"  Custo original:  {total_cost_old:,.0f} R$")
    print(f"  Δ custo:         {total_delta_cost:,.0f} R$")
    print(f"\nEmissões:")
    print(f"  Original:   {total_E_old/1000:,.0f} tCO2")
    print(f"  Nova:       {total_E_new/1000:,.0f} tCO2")
    print(f"  Redução:    {total_delta_E/1000:,.0f} tCO2")
    print(f"\nCMA sistema: {CMA_system:,.2f} R$/tCO2")
    print("="*60)

print(f"✓ Resultados extraídos em {time.time()-t0:.1f}s")

#%%
# ============================================================
# 13) EXPORTAR
# ============================================================
print("\n>>> Exportando resultados...")
out_xlsx = path + f"/otimizacao_substituicao_alpha_{int(ALPHA*100)}pct_tipo.xlsx"

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    if df_flow.empty:
        pd.DataFrame([{
            "msg": "df_flow vazio - verifique viabilidade (alpha, MAX_DIST, oferta)"
        }]).to_excel(writer, sheet_name="erro", index=False)
    else:
        df_summary.to_excel(writer, sheet_name="resumo_sistema", index=False)
        df_plant.to_excel(writer, sheet_name="resultados_por_planta", index=False)
        df_flow.to_excel(writer, sheet_name="fluxos_planta_mun_tipo", index=False)
        
        # Criar planilha com custos por tipo
        df_custos_tipo = pd.DataFrame([
            {
                'Tipo': 'MSW',
                'Custo_materia_prima_R_per_GJ': C_MSW_R_PER_GJ,
                'CAPEX_R_per_GJ': (CAPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'OPEX_R_per_GJ': (OPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'Total_base_R_per_GJ': C_base_tipo['MSW'],
                'GJ_utilizado': by_type.get('MSW', 0.0),
                'Custo_total_R': custo_msw_total
            },
            {
                'Tipo': 'AGRO',
                'Custo_materia_prima_R_per_GJ': C_AGRO_R_PER_GJ,
                'CAPEX_R_per_GJ': (CAPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'OPEX_R_per_GJ': (OPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'Total_base_R_per_GJ': C_base_tipo['AGRO'],
                'GJ_utilizado': by_type.get('AGRO', 0.0),
                'Custo_total_R': custo_agro_total
            },
            {
                'Tipo': 'IW',
                'Custo_materia_prima_R_per_GJ': C_IW_R_PER_GJ,
                'CAPEX_R_per_GJ': (CAPEX_IW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'OPEX_R_per_GJ': (OPEX_IW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO,
                'Total_base_R_per_GJ': C_base_tipo['IW'],
                'GJ_utilizado': by_type.get('IW', 0.0),
                'Custo_total_R': custo_iw_total
            }
        ])
        df_custos_tipo.to_excel(writer, sheet_name="custos_por_tipo", index=False)

print(f"✓ Arquivo Excel exportado: {out_xlsx}")

#%%
# ============================================================
# 14) LOG E CHECKPOINT
# ============================================================
# Salvar log de execução
log_file = path + f"/log_otimizacao_alpha_{int(ALPHA*100)}pct.txt"
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"=== LOG DE EXECUÇÃO ===\n")
    f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Tempo total: {elapsed_solve:.1f}s\n\n")
    
    f.write(f"=== PARÂMETROS ===\n")
    f.write(f"Alpha: {ALPHA}\n")
    f.write(f"MAX_DIST_KM: {MAX_DIST_KM}\n")
    f.write(f"Solver: {solver_name}\n\n")
    
    f.write(f"=== CUSTOS DIFERENCIADOS ===\n")
    f.write(f"MSW - Custo base: {C_base_tipo['MSW']:.2f} R$/GJ\n")
    f.write(f"  - Matéria-prima: {C_MSW_R_PER_GJ:.4f} R$/GJ\n")
    f.write(f"  - CAPEX: {(CAPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ\n")
    f.write(f"  - OPEX: {(OPEX_MSW_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ\n\n")
    f.write(f"AGRO - Custo base: {C_base_tipo['AGRO']:.2f} R$/GJ\n")
    f.write(f"  - Matéria-prima: {C_AGRO_R_PER_GJ:.4f} R$/GJ\n")
    f.write(f"  - CAPEX: {(CAPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ\n")
    f.write(f"  - OPEX: {(OPEX_AGRO_USD_PER_T / LHV_RESIDUE_MJ_PER_KG) * CAMBIO:.2f} R$/GJ\n\n")
    
    f.write(f"Status solver: {result.solver.status}\n")
    f.write(f"Condição término: {result.solver.termination_condition}\n\n")
    
    if not df_flow.empty:
        f.write(f"=== RESULTADOS ===\n")
        f.write(f"GJ substituídos: {total_gj:,.0f}\n")
        f.write(f"  - MSW: {by_type.get('MSW', 0.0):,.0f} GJ ({perc_msw:.1f}%)\n")
        f.write(f"  - AGRO: {by_type.get('AGRO', 0.0):,.0f} GJ ({perc_agro:.1f}%)\n\n")
        f.write(f"Custos:\n")
        f.write(f"  - MSW: {custo_msw_total:,.0f} R$\n")
        f.write(f"  - AGRO: {custo_agro_total:,.0f} R$\n\n")
        f.write(f"CMA sistema: {CMA_system:,.2f} R$/tCO2\n")
        f.write(f"Redução emissões: {total_delta_E/1000:,.0f} tCO2\n")

print(f"✓ Log salvo: {log_file}")

# Salvar checkpoint
checkpoint_file = path + '/checkpoint_otimizacao.pkl'
checkpoint = {
    'dist_km': dist_km,
    'PMT': PMT,
    'c_pmt': c_pmt,
    'Demand': Demand,
    'Supply_type': Supply_type,
    'parametros': {
        'ALPHA': ALPHA,
        'MAX_DIST_KM': MAX_DIST_KM,
        'C_TRAD_R_PER_GJ': C_TRAD_R_PER_GJ,
        'C_base_MSW': C_base_tipo['MSW'],
        'C_base_AGRO': C_base_tipo['AGRO'],
        'C_transp_factor': C_transp_factor
    }
}

with open(checkpoint_file, 'wb') as f:
    pickle.dump(checkpoint, f)

print(f"✓ Checkpoint salvo: {checkpoint_file}")

print("\n" + "="*60)
print("EXECUÇÃO FINALIZADA COM SUCESSO!")
print("="*60)
