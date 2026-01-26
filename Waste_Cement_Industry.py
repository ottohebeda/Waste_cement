# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:23:42 2025

@author: ottoh

"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt  # necessário para ler WKT
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

path = r"C:\Users\otto.hebeda\OneDrive - epe.gov.br\Documentos\Resíduos - projetos\Residuos na produção de cimento\Dados"


#%%
# ================================================================
# 1. Ler shapefile dos municípios
# ================================================================
gdf_mun = gpd.read_file(path+"/BR_Municipios_2024.shp")
gdf_mun["CD_MUN"] = gdf_mun["CD_MUN"].astype(str).str.zfill(7)

#%%
# ================================================================
# 2. Ler os potenciais energéticos (MSW + Agro)
# ================================================================
df_msw = pd.read_csv(path+"/MSW_Energy_potential.csv")
df_agro = pd.read_csv(path+"/AgroWaste_Energy_potential.csv")

df_msw["CD_MUN"] = df_msw["CD_MUN"].astype(str).str.zfill(7)
df_agro["CD_MUN"] = df_agro["CD_MUN"].astype(str).str.zfill(7)

df = df_msw.merge(df_agro, on="CD_MUN", how="outer")

df["potencial_total_GJ"] = (
    df["Energy potential (GJ)"].fillna(0) +
    df["Total (GJ)"].fillna(0)
)

df_final = df[["CD_MUN", "potencial_total_GJ"]].rename(
    columns={"CD_MUN": "CD_MUN"}
)

gdf_merge = gdf_mun.merge(df_final, on="CD_MUN", how="left")

#%%
# ================================================================
# 3. Ler arquivo Fabricas_geo com a coluna "Georreferenciamento"
# ================================================================
df_fab = pd.read_csv(path+"/Fabricas_geo.csv")

# Separar latitude e longitude
df_fab[["lat", "lon"]] = df_fab["Georreferenciamento"].str.split(",", expand=True)

# Converter para float
df_fab["lat"] = df_fab["lat"].astype(float)
df_fab["lon"] = df_fab["lon"].astype(float)

# Criar GeoDataFrame
gdf_fab = gpd.GeoDataFrame(
    df_fab,
    geometry=gpd.points_from_xy(df_fab["lon"], df_fab["lat"]),
    crs="EPSG:4326"
)

# Garantir o mesmo CRS do shapefile
gdf_fab = gdf_fab.to_crs(gdf_mun.crs)


#%%
# ================================================================
# 4. Plotar o mapa final
# ================================================================
fig, ax = plt.subplots(figsize=(12, 10))

gdf_merge.plot(
    column="potencial_total_GJ",
    cmap="viridis",
    legend=True,
    linewidth=0.1,
    edgecolor="black",
    ax=ax
)

# Plotas plantas de cimento
gdf_fab.plot(
    ax=ax,
    color="red",
    markersize=40,
    label="Plantas de cimento"
)

plt.title("Potencial Energético + Localização das Fábricas de Cimento")
plt.legend()
plt.axis("off")
plt.show()

#%%

"""Parameters"""
bag_weight = 0.050 #50 kg
PCS_generico = 20000 #GJ/kt 
gj_to_ktoe = 1/41868
ktoe_to_gj = 41868

Energy_intensity_CP = 3500

"""Importing values - Urban waste"""
Residuos_coletados = pd.read_csv(path+'/Total_residuos_publicos_domiciliares.csv')

Residuos_coletados['Total coletado (kt)'] = Residuos_coletados['Total coletado (t)']/1000
Residuos_coletados['Residuos/habitante'] = Residuos_coletados['Total coletado (t)']/Residuos_coletados['Populacao total']
Residuos_coletados = Residuos_coletados.sort_values('Total coletado (t)')
Residuos_coletados['Total coletado (log kt)'] = np.log10(Residuos_coletados ['Total coletado (kt)'])

"""Importing values - DF with plants, their capacity, their location, the cities within the radius and the amount of waste each city generates"""
DF_plant_waste_by_city= pd.read_csv(path+'/Intersection_radius_cement_plants_R03_v3.csv')
DF_plant_waste_by_city['Cap__Mensal__scs_'] = DF_plant_waste_by_city['Cap__Mensal__scs_'].str.replace(',','')
DF_plant_waste_by_city['Cap__Mensal__scs_'] = DF_plant_waste_by_city['Cap__Mensal__scs_'].str.replace('sacos','')
DF_plant_waste_by_city['Cap__Mensal__scs_'] = DF_plant_waste_by_city['Cap__Mensal__scs_'].str.replace('\n','')
DF_plant_waste_by_city['Cap__Mensal__scs_'] = DF_plant_waste_by_city['Cap__Mensal__scs_'].str.replace('.','')
DF_plant_waste_by_city['Cap__Mensal__scs_'] = DF_plant_waste_by_city['Cap__Mensal__scs_'].astype('int64')
DF_plant_waste_by_city['Capacidade_ins_(t)'] = DF_plant_waste_by_city['Cap__Mensal__scs_']*bag_weight*12 #12 is the number of months
DF_plant_waste_by_city['UF'] = 0
for cidade in DF_plant_waste_by_city.Cidade:
    DF_plant_waste_by_city.loc[DF_plant_waste_by_city.Cidade == cidade,'UF'] = cidade[-2:]

DF_plant_waste_by_city.loc[DF_plant_waste_by_city.Name == 'Cimento Uau - Pains/MG','UF'] = 'MG'
DF_plant_waste_by_city.loc[DF_plant_waste_by_city.Name == "Cimento Forte - Suape/PE",'UF'] = 'PE'


"""Creating a DF with the plants and their capacity"""
w= 0
Capacidade_plantas =pd.DataFrame(index= DF_plant_waste_by_city.Name.unique(), columns= ['Capacidade anual'],data = 0,dtype=float)
for planta in Capacidade_plantas.index:
   if planta != w:
       Capacidade_plantas.loc[planta,'Capacidade anual'] = DF_plant_waste_by_city.loc[DF_plant_waste_by_city.Name == planta,'Capacidade_ins_(t)'].unique()[0]
       Capacidade_plantas.loc[planta,'UF'] = DF_plant_waste_by_city.loc[DF_plant_waste_by_city.Name == planta,'UF'].unique()[0]
       w=planta
   else:
       pass

"""Cement production in each Federal Unity"""
Producao_UF_2019 = pd.read_csv(path+'/producao_cimento_UF_2019.csv', sep = ';') #Ajuste é feito passando o restante da produção para as plantas
Producao_clinquer_2019 = 63000*.71 #kt

"""Production of each plant"""
Producao_plantas = pd.DataFrame(index=Capacidade_plantas.index, columns = ['Producao (kt)','Tipo', 'Producao clinquer (kt)'],data = 0,dtype=float)
w=0
for planta in Producao_plantas.index:
       Producao_plantas.loc[planta,'Producao (kt)'] = float(Capacidade_plantas.loc[planta,'Capacidade anual']/(Capacidade_plantas.groupby('UF').sum().loc[Capacidade_plantas.loc[planta, 'UF']]))*float(Producao_UF_2019.loc[Producao_UF_2019['UF'] == Capacidade_plantas.loc[planta, 'UF'],'Producao (kt) Ajustada'])
       Producao_plantas.loc[planta,'Tipo'] = DF_plant_waste_by_city.loc[DF_plant_waste_by_city['Name'] == planta,'Tipo_Planta'].unique()[0]


for planta in Producao_plantas.index:
    if Producao_plantas.loc[planta,'Tipo'] == 'Fábrica':
        Producao_plantas.loc[planta,'Producao clinquer (kt)'] = Producao_plantas.loc[planta,'Producao (kt)']/Producao_plantas.loc[Producao_plantas['Tipo'] == 'Fábrica','Producao (kt)'].sum()*Producao_clinquer_2019#percentual da produção da planta na aprodução de cimento total
    else:
        Producao_plantas.loc[planta,'Producao clinquer (kt)'] =0

Producao_plantas['Producao (kt)'] =Producao_plantas['Producao (kt)'].astype(float)
Producao_plantas['Producao clinquer (kt)'] =Producao_plantas['Producao clinquer (kt)'].astype(float)       

Producao_plantas.to_excel(path+'/producao_planta_cimento_2022.xlsx')

"""Energy Demand in each plant"""
Energy_demand_plant = Producao_plantas['Producao clinquer (kt)']*Energy_intensity_CP*1
Energy_demand_plant = Energy_demand_plant.to_frame()
Energy_demand_plant= Energy_demand_plant.rename({'Producao clinquer (kt)' :'Energy demand (GJ)'},axis= 1)
Energy_demand_plant['Name'] = Energy_demand_plant.index
#%%
# --- 1) Preparar df_demand (garantir Name + Demand_GJ) ---
df_demand = Energy_demand_plant.copy()

# remover colunas duplicadas
df_demand = df_demand.loc[:, ~df_demand.columns.duplicated()]

# se o nome da planta estiver no index, mover para coluna
if df_demand.index.name is not None and "Name" not in df_demand.columns:
    df_demand = df_demand.reset_index().rename(columns={df_demand.index.name: "Name"})
elif "Name" not in df_demand.columns:
    # se não existe coluna Name, tentar encontrar alguma coluna plausível
    possible = [c for c in df_demand.columns if "name" in c.lower() or "plant" in c.lower()]
    if len(possible) == 1:
        df_demand = df_demand.rename(columns={possible[0]: "Name"})
    else:
        raise ValueError("Não foi possível localizar a coluna com o nome da planta em df_demand.")

# renomear coluna de demanda para Demand_GJ (se presente com outro nome)
if "Energy demand (GJ)" in df_demand.columns:
    df_demand = df_demand.rename(columns={"Energy demand (GJ)": "Demand_GJ"})
elif "Demand_GJ" not in df_demand.columns:
    # tentar detectar coluna plausível de demanda
    poss_d = [c for c in df_demand.columns if "demand" in c.lower() or "gJ" in c.lower()]
    if len(poss_d) == 1:
        df_demand = df_demand.rename(columns={poss_d[0]: "Demand_GJ"})
    else:
        raise ValueError("Não foi possível localizar a coluna de demanda em df_demand.")

# manter apenas Name + Demand_GJ
df_demand = df_demand.loc[:, ["Name", "Demand_GJ"]]

# remover duplicatas de Name (se houver), mantendo a primeira
df_demand = df_demand.drop_duplicates(subset="Name", keep="first").reset_index(drop=True)

# --- 2) Garantir que gdf_fab tem coluna 'Name' para merge ---
# se o nome da planta estiver no index de gdf_fab, mover para coluna
if gdf_fab.index.name is not None and "Name" not in gdf_fab.columns:
    gdf_fab = gdf_fab.reset_index().rename(columns={gdf_fab.index.name: "Name"})
elif "Name" not in gdf_fab.columns:
    # tentar achar coluna plausível
    poss = [c for c in gdf_fab.columns if "name" in c.lower() or "plant" in c.lower()]
    if len(poss) == 1:
        gdf_fab = gdf_fab.rename(columns={poss[0]: "Name"})
    else:
        raise ValueError("Não foi possível localizar a coluna 'Name' em gdf_fab para fazer o merge.")

# --- 3) Remover qualquer coluna antiga Demand_GJ em gdf_fab para evitar duplicação ---
if "Demand_GJ" in gdf_fab.columns:
    gdf_fab = gdf_fab.drop(columns=["Demand_GJ"])

# --- 4) Fazer merge seguro ---
gdf_fab = gdf_fab.merge(df_demand, on="Name", how="left")

# verificar quantas plantas ficaram sem demanda
n_missing = gdf_fab["Demand_GJ"].isna().sum()
if n_missing > 0:
    print(f"Atenção: {n_missing} plantas ficaram sem Demand_GJ após o merge. Verifique correspondência de nomes.")

# --- 5) Reprojetar para CRS em metros (só depois do merge) ---
gdf_fab = gdf_fab.to_crs(5880)
gdf_merge = gdf_merge.to_crs(5880)

# --- 6) CRIAR CÓPIA DA OFERTA DISPONÍVEL (que será reduzida dinamicamente) ---
gdf_supply = gdf_merge.copy()
gdf_supply["oferta_disponivel_GJ"] = gdf_supply["potencial_total_GJ"].copy()

# --- 7) Função MODIFICADA para considerar oferta limitada ---
def find_radius_for_plant(plant, gdf_polygons_supply, step_km=1, max_radius_km=300):
    """
    Retorna o raio mínimo necessário (km) para suprir 100% da demanda da planta,
    considerando apenas a oferta disponível.
    Também retorna quais municípios foram usados e quanto foi consumido de cada um.
    """
    demand = plant.get("Demand_GJ", None)
    if pd.isna(demand) or demand is None:
        name = plant.get("Name", None)
        if name is not None and name in df_demand["Name"].values:
            demand = float(df_demand.loc[df_demand["Name"] == name, "Demand_GJ"].iloc[0])
        else:
            return None, []

    radius = step_km * 1000  # metros
    
    while radius <= max_radius_km * 1000:
        buffer = plant.geometry.buffer(radius)
        
        # municípios dentro do buffer
        subset = gdf_polygons_supply[gdf_polygons_supply.intersects(buffer)].copy()
        
        # oferta disponível total no raio
        total_disponivel = subset["oferta_disponivel_GJ"].sum()
        
        if total_disponivel >= demand:
            # Alocar a demanda proporcionalmente à oferta disponível de cada município
            # ou simplesmente consumir sequencialmente até preencher a demanda
            
            # Vamos alocar sequencialmente (pode ordenar por proximidade se preferir)
            demanda_restante = demand
            alocacoes = []
            
            for idx, mun in subset.iterrows():
                disponivel = mun["oferta_disponivel_GJ"]
                
                if demanda_restante <= 0:
                    break
                    
                if disponivel > 0:
                    consumido = min(disponivel, demanda_restante)
                    alocacoes.append({
                        'index': idx,
                        'consumido_GJ': consumido
                    })
                    demanda_restante -= consumido
            
            return radius / 1000, alocacoes  # retorna km e lista de alocações
        
        radius += step_km * 1000

    return None, []

# --- 8) Aplicar para todas as plantas, ORDENANDO por algum critério ---
# Você pode ordenar por demanda (maior primeiro), ou por outra prioridade
# Aqui vamos ordenar por demanda decrescente (plantas maiores têm prioridade)

gdf_fab = gdf_fab[gdf_fab["Demand_GJ"].fillna(0) > 0].copy()
gdf_fab = gdf_fab.sort_values("Demand_GJ", ascending=False).reset_index(drop=True)

results = []

for idx, plant in gdf_fab.iterrows():
    r, alocacoes = find_radius_for_plant(plant, gdf_supply)
    
    # Registrar resultado
    demanda_atendida = sum([a['consumido_GJ'] for a in alocacoes])
    results.append({
        "Name": plant["Name"],
        "Demand_GJ": plant["Demand_GJ"],
        "radius_km": r,
        "demanda_atendida_GJ": demanda_atendida,
        "percentual_atendido": (demanda_atendida / plant["Demand_GJ"] * 100) if plant["Demand_GJ"] > 0 else 0
    })
    
    # ATUALIZAR a oferta disponível (reduzir conforme foi consumido)
    for aloc in alocacoes:
        gdf_supply.loc[aloc['index'], 'oferta_disponivel_GJ'] -= aloc['consumido_GJ']
    
    # Log de progresso
    if r is not None:
        print(f"Planta '{plant['Name']}': raio {r:.1f} km, atendida {demanda_atendida:.0f}/{plant['Demand_GJ']:.0f} GJ ({demanda_atendida/plant['Demand_GJ']*100:.1f}%)")
    else:
        print(f"Planta '{plant['Name']}': NÃO foi possível atender a demanda de {plant['Demand_GJ']:.0f} GJ mesmo com raio máximo")

df_radius = pd.DataFrame(results)

# --- 9) Verificar oferta final ---
print(f"\n=== RESUMO ===")
print(f"Oferta original total: {gdf_merge['potencial_total_GJ'].sum():.0f} GJ")
print(f"Oferta restante: {gdf_supply['oferta_disponivel_GJ'].sum():.0f} GJ")
print(f"Oferta consumida: {gdf_merge['potencial_total_GJ'].sum() - gdf_supply['oferta_disponivel_GJ'].sum():.0f} GJ")
print(f"Demanda total das plantas: {df_radius['Demand_GJ'].sum():.0f} GJ")
print(f"Demanda atendida: {df_radius['demanda_atendida_GJ'].sum():.0f} GJ")
#%%
import pandas as pd
import numpy as np

# ============================================================
# Parâmetros fixos
# ============================================================
CAMBIO = 5.5  # R$ por US$
COST_USD_PER_GJ_PER_KM = 0.0066  # US$/GJ/km

# Conversões e custos por GJ
GJ_PER_TEP = 41.868
C_trad = 466.0 / GJ_PER_TEP         # R$/GJ (combustível tradicional)
C_resid = 500.0 / GJ_PER_TEP        # R$/GJ (combustível de resíduo)
C_transp_factor = COST_USD_PER_GJ_PER_KM * CAMBIO  # R$/GJ/km

print(f"C_trad = {C_trad:.6f} R$/GJ")
print(f"C_resid = {C_resid:.6f} R$/GJ")
print(f"C_transp_factor = {C_transp_factor:.6f} R$/GJ/km")

# ============================================================
# Parâmetros de transporte e emissões
# ============================================================
truck_payload_t = 20.0               # capacidade útil do caminhão (t)
LHV_residue_MJ_per_kg = 10.0         # PCI do resíduo (MJ/kg)
diesel_L_per_km = 0.30               # consumo do caminhão (L/km)
EF_diesel_kgCO2_per_L = 2.67         # fator de emissão diesel (kgCO2/L)

# Energia transportada por viagem (GJ)
GJ_per_truck = (truck_payload_t * 1000 * LHV_residue_MJ_per_kg) / 1000  # MJ->GJ

# Emissões na combustão
EF_trad_kgCO2_per_GJ = 92.8
EF_resid_ratio = 0.05
EF_resid_kgCO2_per_GJ = EF_trad_kgCO2_per_GJ * EF_resid_ratio

# ============================================================
# CAPEX e OPEX do combustível de resíduo (NOVO)
# ============================================================
CAPEX_USD_PER_T = 30.0   # US$/t combustível produzido
OPEX_USD_PER_T = 0  # US$/t combustível produzido

GJ_PER_TON_RESID = LHV_residue_MJ_per_kg  # GJ/t (1 t => 1000 kg)

CAPEX_R_per_GJ = (CAPEX_USD_PER_T / GJ_PER_TON_RESID) * CAMBIO
OPEX_R_per_GJ = (OPEX_USD_PER_T / GJ_PER_TON_RESID) * CAMBIO
CAPEX_OPEX_R_per_GJ = CAPEX_R_per_GJ + OPEX_R_per_GJ

print(f"CAPEX_R_per_GJ = {CAPEX_R_per_GJ:.2f} R$/GJ")
print(f"OPEX_R_per_GJ = {OPEX_R_per_GJ:.2f} R$/GJ")
print(f"CAPEX+OPEX = {CAPEX_OPEX_R_per_GJ:.2f} R$/GJ")

# ============================================================
# Preparar dados
# ============================================================
df_plants = gdf_fab.copy()
if hasattr(df_plants, "geometry"):
    df_plants = df_plants.drop(columns=["geometry"], errors="ignore")
df_plants = df_plants.loc[:, ~df_plants.columns.duplicated()]

if "Name" not in df_plants.columns:
    raise ValueError("gdf_fab precisa ter coluna 'Name' antes deste cálculo.")
if "Demand_GJ" not in df_plants.columns:
    raise ValueError("gdf_fab precisa ter coluna 'Demand_GJ' antes deste cálculo.")

# garantir que df_radius NÃO tenha Demand_GJ
df_radius_clean = df_radius.drop(columns=["Demand_GJ"], errors="ignore")

df_cost = df_radius_clean.merge(
    df_plants[["Name", "Demand_GJ"]],
    on="Name",
    how="left"
)

# df_cost["Demand_GJ"] = df_cost["Demand_GJ_y"].combine_first(df_cost["Demand_GJ_x"])

# Remover colunas duplicadas
# df_cost = df_cost.drop(columns=["Demand_GJ_x", "Demand_GJ_y"])
# Missing demand?
if df_cost["Demand_GJ"].isna().sum() > 0:
    print("Atenção: plantas com Demand_GJ ausente.")

df_cost["radius_km"] = pd.to_numeric(df_cost["radius_km"], errors="coerce")

# ============================================================
# Cálculos de custo originais
# ============================================================
# custo de transporte por GJ (R$/GJ)
df_cost["C_transp_R_per_GJ"] = df_cost["radius_km"] * C_transp_factor

# custo do combustível de resíduo entregue (sem CAPEX/OPEX ainda)
df_cost["C_resid_total_R_per_GJ"] = C_resid + df_cost["C_transp_R_per_GJ"]

# ============================================================
# >>> NOVO: adicionar CAPEX+OPEX por GJ ao custo total do resíduo
# ============================================================
df_cost["CAPEX_R_per_GJ"] = CAPEX_R_per_GJ
df_cost["OPEX_R_per_GJ"] = OPEX_R_per_GJ
df_cost["CAPEX_OPEX_R_per_GJ"] = CAPEX_OPEX_R_per_GJ

# Custo TOTAL do resíduo (combustível + transporte + CAPEX + OPEX)
df_cost["C_total_resid_R_per_GJ"] = (
    df_cost["C_resid_total_R_per_GJ"] + df_cost["CAPEX_OPEX_R_per_GJ"]
)

# diferença unitária (resíduo - tradicional) R$/GJ
df_cost["Delta_R_per_GJ"] = df_cost["C_total_resid_R_per_GJ"] - C_trad

# custo anual com resíduo (R$/ano) – total
df_cost["Custo_anual_resid_R$"] = df_cost["C_total_resid_R_per_GJ"] * df_cost["Demand_GJ"]

# custo anual atual com combustível tradicional (R$/ano)
df_cost["Custo_anual_trad_R$"] = C_trad * df_cost["Demand_GJ"]

# economia anual (tradicional - resíduo total): positivo = economia; negativo = aumento de custo
df_cost["Economia_anual_R$"] = df_cost["Custo_anual_trad_R$"] - df_cost["Custo_anual_resid_R$"]

# percentual de mudança (resíduo vs tradicional)
df_cost["Perc_change_%"] = 100.0 * (df_cost["C_total_resid_R_per_GJ"] / C_trad - 1.0)

# ============================================================
# Viagens, distâncias e emissões (mantendo o que já tínhamos)
# ============================================================
df_cost["GJ_per_truck"] = GJ_per_truck
df_cost["n_trips"] = np.ceil(df_cost["Demand_GJ"] / df_cost["GJ_per_truck"])
df_cost["n_trips"] = df_cost["n_trips"].replace([np.inf, -np.inf], np.nan)

df_cost["dist_per_trip_km"] = 2 * df_cost["radius_km"]
df_cost["total_dist_km"] = df_cost["dist_per_trip_km"] * df_cost["n_trips"]

df_cost["litros_diesel"] = df_cost["total_dist_km"] * diesel_L_per_km
df_cost["E_transport_kgCO2"] = df_cost["litros_diesel"] * EF_diesel_kgCO2_per_L

df_cost["E_trad_comb_kgCO2"] = df_cost["Demand_GJ"] * EF_trad_kgCO2_per_GJ
df_cost["E_resid_comb_kgCO2"] = df_cost["Demand_GJ"] * EF_resid_kgCO2_per_GJ
df_cost["E_total_resid_kgCO2"] = df_cost["E_resid_comb_kgCO2"] + df_cost["E_transport_kgCO2"]

df_cost["Saving_kgCO2"] = df_cost["E_trad_comb_kgCO2"] - df_cost["E_total_resid_kgCO2"]
df_cost["Saving_percent"] = 100 * df_cost["Saving_kgCO2"] / df_cost["E_trad_comb_kgCO2"]

# ============================================================
# Colunas finais
# ============================================================
cols_out = [
    "Name", "radius_km", "Demand_GJ",
    "C_transp_R_per_GJ", "C_resid_total_R_per_GJ",
    "CAPEX_R_per_GJ", "OPEX_R_per_GJ", "CAPEX_OPEX_R_per_GJ",
    "C_total_resid_R_per_GJ", "Delta_R_per_GJ",
    "Custo_anual_trad_R$", "Custo_anual_resid_R$", "Economia_anual_R$", "Perc_change_%",
    "GJ_per_truck", "n_trips", "dist_per_trip_km", "total_dist_km",
    "litros_diesel", "E_transport_kgCO2",
    "E_trad_comb_kgCO2", "E_resid_comb_kgCO2", "E_total_resid_kgCO2",
    "Saving_kgCO2", "Saving_percent"
]

df_cost = df_cost[cols_out]

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(df_cost.head())

df_cost.to_csv(path+"/costs_residue_substitution_per_plant.csv", index=False)


#%%

# --- Fatores de emissão ---
EF_trad_kgCO2_per_GJ = 93                     # combustível tradicional (coque)
EF_resid_kgCO2_per_GJ = 0.05 * EF_trad_kgCO2_per_GJ  # combustível alternativo = 5% do tradicional

EF_transporte_kgCO2_km = 2.7                  # caminhão diesel (aprox.)

# Limpar duplicidades antigas antes de novo merge
cols = gdf_fab.columns

if "Demand_GJ_x" in cols or "Demand_GJ_y" in cols:
    gdf_fab["Demand_GJ"] = (
        gdf_fab.get("Demand_GJ_y", np.nan)
        .combine_first(gdf_fab.get("Demand_GJ_x", np.nan))
    )
    gdf_fab = gdf_fab.drop(
        columns=[c for c in ["Demand_GJ_x", "Demand_GJ_y"] if c in cols]
    )

gdf_fab = gdf_fab.merge(
    df_radius[["Name", "radius_km"]],
    on="Name",
    how="left"
)

# Criar Demand_GJ final escolhendo a coluna correta

# --- Emissões combustíveis ---
gdf_fab["Emissoes_trad_kgCO2"] = (
    gdf_fab["Demand_GJ"] * EF_trad_kgCO2_per_GJ
)

gdf_fab["Emissoes_resid_kgCO2"] = (
    gdf_fab["Demand_GJ"] * EF_resid_kgCO2_per_GJ
)

# --- Emissões de transporte ---
gdf_fab["Emissoes_transporte_kgCO2"] = (
    gdf_fab["radius_km"] * EF_transporte_kgCO2_km
)

# --- Emissões totais após substituição ---
gdf_fab["Emissoes_totais_substituicao_kgCO2"] = (
    gdf_fab["Emissoes_resid_kgCO2"] + gdf_fab["Emissoes_transporte_kgCO2"]
)

# --- Redução total de emissões ---
gdf_fab["Reducao_kgCO2"] = (
    gdf_fab["Emissoes_trad_kgCO2"] - gdf_fab["Emissoes_totais_substituicao_kgCO2"]
)

#%%
# Unir o raio calculado ao gdf_fab
df_final = gdf_fab.merge(df_radius, on="Name", how="left")
df_final = gdf_fab.merge(df_cost,on="Name",how="left")

output_path = path+"/resultados_por_planta.xlsx"

df_final.to_excel(output_path, index=False)

print(f"Arquivo exportado com sucesso para: {output_path}")
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, Constraint, SolverFactory, value
from amplpy import modules

# ============================================================
# 0) PARAMETROS DO PROBLEMA
# ============================================================
alpha = 0.5  # percentual de substituição (ex: 30%) -> ajuste aqui

# Custos unitários (R$/GJ) - use os seus já calculados
# C_resid = 500/41.868  (R$/GJ)
# CAPEX_OPEX_R_per_GJ já calculado no seu script
# C_transp_factor = 0.0066*CAMBIO  (R$/GJ/km)

# Aqui vou assumir que essas variáveis já existem no seu ambiente:
# C_resid, CAPEX_OPEX_R_per_GJ, C_transp_factor

C_base_R_per_GJ = float(C_resid) + float(CAPEX_OPEX_R_per_GJ)  # custo do resíduo "na porta", sem transporte

# ============================================================
# 1) PREPARAR DADOS (PLANTAS E MUNICIPIOS)
# ============================================================
# Garantir CRS métrico
if str(gdf_fab.crs) != "EPSG:5880":
    gdf_fab_m = gdf_fab.to_crs(5880).copy()
else:
    gdf_fab_m = gdf_fab.copy()

if str(gdf_merge.crs) != "EPSG:5880":
    gdf_mun_m = gdf_merge.to_crs(5880).copy()
else:
    gdf_mun_m = gdf_merge.copy()

# filtrar plantas com demanda >0
gdf_fab_m = gdf_fab_m[gdf_fab_m["Demand_GJ"].fillna(0) > 0].copy()

# oferta municipal
gdf_mun_m["Supply_GJ"] = gdf_mun_m["potencial_total_GJ"].fillna(0)

# reduzir tamanho do problema (opcional mas recomendado):
# manter só municipios com oferta > 0
gdf_mun_m = gdf_mun_m[gdf_mun_m["Supply_GJ"] > 0].copy()

# usar centróide do município para distância (simplificação padrão)
gdf_mun_m["centroid"] = gdf_mun_m.geometry.centroid

# criar tabelas base
plants = gdf_fab_m[["Name", "Demand_GJ", "geometry"]].copy()
mun = gdf_mun_m[["CD_MUN", "Supply_GJ", "centroid"]].copy()

plants["Name"] = plants["Name"].astype(str)
mun["CD_MUN"] = mun["CD_MUN"].astype(str).str.zfill(7)

# dicionários de demanda e oferta
Demand = plants.set_index("Name")["Demand_GJ"].to_dict()
Supply = mun.set_index("CD_MUN")["Supply_GJ"].to_dict()

P = list(Demand.keys())
M = list(Supply.keys())

# ============================================================
# 2) MATRIZ DE DISTÂNCIAS (km) ENTRE PLANTA E MUNICÍPIO
# ============================================================
# cálculo vetorizado simples (pode ser pesado se M for grande; depois te mostro otimização)
plant_geom = plants.set_index("Name")["geometry"]
mun_cent = mun.set_index("CD_MUN")["centroid"]

dist_km = {}
for p in P:
    gp = plant_geom.loc[p]
    # distâncias (m) para todos municípios
    d = mun_cent.distance(gp) / 1000.0  # km
    for m in M:
        dist_km[(p, m)] = float(d.loc[m])

# custo unitário total por par (R$/GJ)
c_pm = {(p, m): C_base_R_per_GJ + C_transp_factor * dist_km[(p, m)] for p in P for m in M}

# ============================================================
# 3) MODELO PYOMO (LP)
# ============================================================
model = ConcreteModel()

# Variável: x[p,m] = GJ de resíduo do município m usado na planta p
model.x = Var(P, M, domain=NonNegativeReals)

# Objetivo: minimizar custo total
def obj_rule(mod):
    return sum(mod.x[p, m] * c_pm[(p, m)] for p in P for m in M)

model.obj = Objective(rule=obj_rule, sense=1)  # 1 = minimize

# Restrição: substituição por planta (igualdade)
def plant_sub_rule(mod, p):
    return sum(mod.x[p, m] for m in M) == alpha * Demand[p]

model.plant_sub = Constraint(P, rule=plant_sub_rule)

# Restrição: oferta municipal
def supply_rule(mod, m):
    return sum(mod.x[p, m] for p in P) <= Supply[m]

model.supply = Constraint(M, rule=supply_rule)

# ============================================================
# 4) RESOLVER
# ============================================================
# Tente primeiro glpk ou cbc:
# solver = SolverFactory("glpk")
# solver = SolverFactory("cbc")
# solver = SolverFactory('ipopt')

solver_name = "ipopt"  # "highs", "cbc",  "couenne", "bonmin", "ipopt", "scip", or "gcg".
solver = SolverFactory(solver_name+"nl", executable=modules.find(solver_name), solve_io="nl")

result = solver.solve(model, tee=True)

# ============================================================
# 5) EXTRAIR RESULTADOS
# ============================================================
# fluxos positivos
rows = []
for p in P:
    for m in M:
        xval = value(model.x[p, m])
        if xval is not None and xval > 1e-6:
            rows.append({
                "Name": p,
                "CD_MUN": m,
                "GJ_alocado": xval,
                "dist_km": dist_km[(p, m)],
                "custo_unit_R_per_GJ": c_pm[(p, m)],
                "custo_total_R": xval * c_pm[(p, m)]
            })

df_flow = pd.DataFrame(rows)

# custo por planta
df_cost_opt = (
    df_flow.groupby("Name")
    .agg(
        GJ_substituido=("GJ_alocado", "sum"),
        custo_total_R=("custo_total_R", "sum"),
        dist_media_km=("dist_km", lambda s: np.average(s, weights=df_flow.loc[s.index, "GJ_alocado"]))
    )
    .reset_index()
)

df_cost_opt["custo_medio_R_per_GJ"] = df_cost_opt["custo_total_R"] / df_cost_opt["GJ_substituido"]

# custo total sistema
total_cost = df_flow["custo_total_R"].sum()
total_GJ = df_flow["GJ_alocado"].sum()

print("\n=== RESULTADO OTIMIZAÇÃO ===")
print(f"Substituição alpha = {alpha:.2%}")
print(f"GJ substituídos total = {total_GJ:,.0f}")
print(f"Custo total (R$) = {total_cost:,.0f}")
print(f"Custo médio (R$/GJ) = {total_cost/total_GJ:,.2f}")

# ============================================================
# 6) EXPORTAR
# ============================================================
out_xlsx = path + f"/otimizacao_substituicao_alpha_{int(alpha*100)}pct.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df_cost_opt.to_excel(writer, sheet_name="custos_por_planta", index=False)
    df_flow.to_excel(writer, sheet_name="fluxos_planta_mun", index=False)

print(f"\nArquivo exportado: {out_xlsx}")
#%%

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import folium

# =========================
# 1) Preparar geometrias em lat/lon (EPSG:4326)
# =========================
gdf_fab_ll = gdf_fab.to_crs(4326).copy()
gdf_mun_ll = gdf_merge.to_crs(4326).copy()

gdf_mun_ll["CD_MUN"] = gdf_mun_ll["CD_MUN"].astype(str).str.zfill(7)

# centróide dos municípios (em lat/lon ok para visualização)
gdf_mun_ll["centroid"] = gdf_mun_ll.geometry.centroid

# dicionários de geometria
plant_pt = gdf_fab_ll.set_index("Name").geometry.to_dict()
mun_pt   = gdf_mun_ll.set_index("CD_MUN")["centroid"].to_dict()

# =========================
# 2) (Opcional) reduzir número de linhas para não ficar pesado
#    Ex.: manter só fluxos acima de um limiar
# =========================
df_lines = df_flow.copy()
df_lines["CD_MUN"] = df_lines["CD_MUN"].astype(str).str.zfill(7)

# exemplo: manter só fluxos >= 1% do total ou >= X GJ
threshold = max(df_lines["GJ_alocado"].sum() * 0.0001, 0)  # 1% do total
df_lines = df_lines[df_lines["GJ_alocado"] >= threshold].copy()

# (alternativa) manter top N por planta:
# df_lines = (df_lines.sort_values("GJ_alocado", ascending=False)
#                    .groupby("Name").head(10).reset_index(drop=True))

# =========================
# 3) Criar GeoDataFrame de linhas
# =========================
geoms = []
rows = []

for _, r in df_lines.iterrows():
    p = r["Name"]
    m = r["CD_MUN"]

    if (p not in plant_pt) or (m not in mun_pt):
        continue

    line = LineString([mun_pt[m], plant_pt[p]])
    geoms.append(line)

    rows.append({
        "Name": p,
        "CD_MUN": m,
        "GJ_alocado": float(r["GJ_alocado"]),
        "dist_km": float(r["dist_km"]) if "dist_km" in r else np.nan,
        "custo_total_R": float(r["custo_total_R"]) if "custo_total_R" in r else np.nan
    })

gdf_lines = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")

# =========================
# 4) Mapa Folium
# =========================
# centro do mapa
center = [gdf_fab_ll.geometry.y.mean(), gdf_fab_ll.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

# adicionar plantas (pontos)
for _, row in gdf_fab_ll.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        tooltip=row["Name"],
        fill=True
    ).add_to(m)

# largura das linhas proporcional ao fluxo
max_gj = gdf_lines["GJ_alocado"].max() if len(gdf_lines) else 1.0

def line_style(feat):
    gj = feat["properties"]["GJ_alocado"]
    w = 1 + 6 * (gj / max_gj)  # escala simples
    return {"weight": w, "opacity": 0.7}

tooltip = folium.GeoJsonTooltip(
    fields=["Name", "CD_MUN", "GJ_alocado", "dist_km", "custo_total_R"],
    aliases=["Planta:", "Município:", "GJ alocado:", "Distância (km):", "Custo total (R$):"],
    localize=True
)

folium.GeoJson(
    gdf_lines,
    name="Fluxos (município → planta)",
    style_function=line_style,
    tooltip=tooltip
).add_to(m)

folium.LayerControl().add_to(m)

# salvar
out_html = path + "/mapa_fluxos_municipio_planta.html"
m.save(out_html)

out_html
