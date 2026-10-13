"""
SPK Risiko Diabetes - K-Nearest Neighbors (Streamlit)
Tema: Sunset Gold & Charcoal (Fixed Number Formatting)
Jalankan:  python -m streamlit run app.py
"""
import json
import math
import os

import pandas as pd
import streamlit as st

FILE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_diabetes.json")
FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
DEFAULT = [
    ["P01", 25, 21.5, 90, 110, 0, 0], ["P02", 30, 23.1, 95, 115, 0, 0],
    ["P03", 35, 25.0, 105, 120, 0, 0], ["P04", 40, 27.5, 120, 125, 1, 1],
    ["P05", 45, 30.2, 140, 135, 1, 1], ["P06", 50, 31.5, 155, 140, 1, 1],
    ["P07", 55, 33.0, 165, 145, 1, 1], ["P08", 28, 22.5, 92, 112, 0, 0],
    ["P09", 38, 26.8, 115, 122, 0, 0], ["P10", 48, 29.5, 145, 130, 1, 1],
]


# ---------- data ----------
def muat_data():
    try:
        with open(FILE_DATA, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return [r[:] for r in DEFAULT]


def simpan_data(data):
    try:
        with open(FILE_DATA, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


# ---------- algoritma KNN ----------
def hitung_knn(data, x, k):
    mn = [min(r[j + 1] for r in data) for j in range(5)]
    mx = [max(r[j + 1] for r in data) for j in range(5)]

    def norm(v, j):
        return 0.0 if mx[j] == mn[j] else (v - mn[j]) / (mx[j] - mn[j])

    xn = [norm(v, j) for j, v in enumerate(x)]
    rows = []
    for r in data:
        fitur = r[1:6]
        fn = [norm(v, j) for j, v in enumerate(fitur)]
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(fn, xn)))
        rows.append({"id": r[0], "f": fitur, "fn": fn, "d": d, "kelas": r[6]})
    rows.sort(key=lambda r: r["d"])
    nb = rows[:k]
    tinggi = sum(1 for r in nb if r["kelas"] == 1)
    rendah = k - tinggi
    hasil = nb[0]["kelas"] if tinggi == rendah else (1 if tinggi > rendah else 0)
    return {"mn": mn, "mx": mx, "xn": xn, "rows": rows, "nb": nb,
            "tinggi": tinggi, "rendah": rendah, "hasil": hasil}


label = lambda c: "Tinggi" if c == 1 else "Rendah"


# ---------- helper untuk format & center alignment tabel ----------
def format_dan_center(df_styler):
    return df_styler.set_properties(**{'text-align': 'center'})


# ---------- tampilan (Sunset Gold & Charcoal) ----------
st.set_page_config(page_title="SPK Risiko Diabetes - KNN", page_icon="✨", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #fafaf9;
    color: #292524;
}
.hero {
    background: linear-gradient(135deg, #d97706, #b45309);
    color: white;
    padding: 30px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px rgba(217, 119, 6, 0.2);
}
.hero h1 { margin: 0; font-size: 2.2rem; color: white; font-weight: 700; }
.hero p { margin: 8px 0 0; color: #fef3c7; font-size: 1.05rem; }

.hasil {
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    color: white;
    margin-top: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.hasil.t { background: linear-gradient(135deg, #b45309, #92400e); }
.hasil.r { background: linear-gradient(135deg, #374151, #1f2937); }
.hasil h2 { color: white; margin: 0 0 8px; font-size: 1.5rem; }

.stButton>button, .stFormSubmitButton>button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #d97706, #b45309) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 12px rgba(217, 119, 6, 0.25) !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background: linear-gradient(135deg, #b45309, #78350f) !important;
    box-shadow: 0 6px 16px rgba(217, 119, 6, 0.4) !important;
    transform: translateY(-2px);
}

div[data-testid="stDataFrame"] {
    display: flex;
    justify-content: center;
    width: 100%;
}
div[data-testid="stDataFrame"] table {
    margin: 0 auto;
}
div[data-testid="stDataFrame"] th {
    text-align: center !important;
    background-color: #fef3c7 !important;
    color: #92400e !important;
}
div[data-testid="stDataFrame"] th div {
    justify-content: center !important;
    text-align: center !important;
}
div[data-testid="stDataFrame"] td {
    text-align: center !important;
}

h2, h3 {
    color: #92400e;
    font-weight: 600;
}
</style>

<div class="hero">
    <h1>✨ Sistem Pendukung Keputusan Risiko Diabetes</h1>
    <p>Analisis Klinis Berbasis Algoritme K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = muat_data()
data = st.session_state.data

tab1, tab2 = st.tabs(["🔍 Prediksi Pasien", "📋 Kelola Dataset Latih"])

# ===== tab prediksi =====
with tab1:
    st.subheader("1. Input Data Pasien Baru")
    contoh = st.checkbox("Isi dengan contoh pasien baru dari soal (Usia: 42, BMI: 28.5, Glukosa: 130, TD: 128, Riwayat: 1)")
    
    with st.form("form_pasien"):
        c1, c2, c3, c4 = st.columns(4)
        nama = c1.text_input("Nama Pasien", value="Pasien Baru" if contoh else "")
        usia = c2.number_input("Usia (tahun)", min_value=1, step=1, value=42 if contoh else None, format="%d")
        bmi = c3.number_input("BMI", min_value=1.0, step=0.1, value=28.5 if contoh else None, format="%.1f")
        glu = c4.number_input("Glukosa (mg/dL)", min_value=1, step=1, value=130 if contoh else None, format="%d")
        
        c5, c6, c7 = st.columns(3)
        td = c5.number_input("Tekanan Darah (mmHg)", min_value=1, step=1, value=128 if contoh else None, format="%d")
        rk = c6.selectbox("Riwayat Diabetes Keluarga", ["0 - Tidak ada", "1 - Ada"],
                          index=1 if contoh else None, placeholder="Pilih Riwayat")
        k = c7.number_input("Nilai K", min_value=1, max_value=len(data), step=1, value=min(3, len(data)), format="%d")
        
        st.markdown("<br>", unsafe_allow_html=True)
        kirim = st.form_submit_button("Hitung Klasifikasi KNN", type="primary")

    if kirim:
        if not nama.strip() or rk is None or None in (usia, bmi, glu, td):
            st.warning("⚠️ Mohon lengkapi semua kolom input data pasien terlebih dahulu.")
        else:
            x = [float(usia), float(bmi), float(glu), float(td), int(rk[0])]
            h = hitung_knn(data, x, int(k))
            k_val = int(k)

            st.markdown("---")
            st.subheader("2. Tahapan Perhitungan Algoritme")
            
            st.markdown("**Langkah 1 - Data Pasien Baru**")
            df_pasien = pd.DataFrame([[nama, int(usia), f"{float(bmi):.1f}", int(glu), int(td), int(rk[0])]], 
                                     columns=["Nama"] + FITUR)
            st.dataframe(format_dan_center(df_pasien.style), hide_index=True, use_container_width=True)

           st.markdown("**Langkah 2 - Normalisasi Min-Max**")
            st.latex(r"x' = \frac{x - \min}{\max - \min}")
            
            norm_data = []
            for idx, f_name in enumerate(FITUR):
                mn_val = h["mn"][idx]
                mx_val = h["mx"][idx]
                p_val = x[idx]
                
                if f_name == "BMI":
                    mn_str = f"{mn_val:.1f}"
                    mx_str = f"{mx_val:.1f}"
                    p_str = f"{p_val:.1f}"
                else:
                    mn_str = f"{int(mn_val)}"
                    mx_str = f"{int(mx_val)}"
                    p_str = f"{int(p_val)}"
                
                norm_data.append({
                    "Atribut": f_name,
                    "Min": mn_str,
                    "Max": mx_str,
                    "Nilai Pasien": p_str,
                    "Hasil Normalisasi": f"{h['xn'][idx]:.5f}"
                })
            
            df_norm = pd.DataFrame(norm_data)
            st.dataframe(format_dan_center(df_norm.style), hide_index=True, use_container_width=True)

            st.markdown("**Langkah 3 - Penghitungan Jarak Euclidean**")
            st.latex(r"d = \sqrt{\sum (x'_{pasien} - x'_{latih})^2}")
            r0 = h["rows"][0]
            suku = " + ".join(f"({a:.3f}-{b:.3f})^2" for a, b in zip(h["xn"], r0["fn"]))
            st.caption(f"Contoh Perhitungan Jarak {r0['id']}: d = √({suku}) = {r0['d']:.4f}")
            
            df_jarak = pd.DataFrame(
                [[i + 1, r["id"], int(r["f"][0]), f"{r['f'][1]:.1f}", int(r["f"][2]), int(r["f"][3]), int(r["f"][4]), round(r["d"], 4), label(r["kelas"])]
                 for i, r in enumerate(h["rows"])],
                columns=["Peringkat", "Pasien"] + FITUR + ["Jarak", "Kelas"]
            )
            
            warna_baris = lambda row: ["background-color: #fef3c7; font-weight: bold; color: #92400e;" if row["Peringkat"] <= k_val else "" for _ in row]
            df_jarak_styled = df_jarak.style.apply(warna_baris, axis=1)
            st.dataframe(format_dan_center(df_jarak_styled), hide_index=True, use_container_width=True)
            st.caption(f"Baris dengan latar belakang emas lembut adalah {k_val} tetangga terdekat.")

            st.markdown(f"**Langkah 4 - Voting K = {k_val} Tetangga Terdekat**")
            st.write("Tetangga Terpilih: " + ", ".join(f"<b>{r['id']}</b> ({label(r['kelas'])})" for r in h["nb"]), unsafe_allow_html=True)
            
            m1, m2 = st.columns(2)
            m1.metric("Suara Risiko Tinggi", f"{h['tinggi']} Suara")
            m2.metric("Suara Risiko Rendah", f"{h['rendah']} Suara")
            
            if h["tinggi"] == h["rendah"]:
                st.info("ℹ️ Jumlah suara seimbang, kelas akhir ditentukan berdasarkan tetangga terdekat dengan jarak minimum.")

            st.markdown("---")
            st.subheader("3. Kesimpulan Klasifikasi")
            t_status = h["hasil"] == 1
            st.markdown(
                f"""<div class="hasil {'t' if t_status else 'r'}">
                <h2>{'⚠️ RISIKO DIABETES TINGGI' if t_status else '✅ RISIKO DIABETES RENDAH'}</h2>
                Berdasarkan {k_val} tetangga terdekat ({h['tinggi']} Tinggi, {h['rendah']} Rendah),<br>
                pasien <b>{nama}</b> ({int(usia)} tahun) diklasifikasikan ke dalam kategori<br>
                <b>{'Risiko Tinggi (Kelas 1)' if t_status else 'Risiko Rendah (Kelas 0)'}</b>.<br><br>
                <i style="font-size: 0.9rem;">{'Rekomendasi: Segera lakukan pemeriksaan medis lanjutan dan konsultasi dokter.' if t_status else 'Rekomendasi: Pertahankan pola hidup sehat dan lakukan evaluasi berkala.'}</i>
                </div>""",
                unsafe_allow_html=True
            )

# ===== tab dataset =====
with tab2:
    st.subheader("Dataset Latih Aktif")
    df_latih = pd.DataFrame(
        [[r[0], int(r[1]), f"{r[2]:.1f}", int(r[3]), int(r[4]), int(r[5]), label(r[6])] for r in data],
        columns=["Pasien"] + FITUR + ["Kelas"]
    )
    st.dataframe(format_dan_center(df_latih.style), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Tambah Data Latih Baru")
    with st.form("form_tambah", clear_on_submit=True):
        a1, a2, a3, a4 = st.columns(4)
        n_id = a1.text_input("Kode Pasien (Cth: P11)")
        n_usia = a2.number_input("Usia", min_value=1, step=1, value=None, format="%d")
        n_bmi = a3.number_input("BMI", min_value=1.0, step=0.1, value=None, format="%.1f")
        n_glu = a4.number_input("Glukosa", min_value=1, step=1, value=None, format="%d")
        
        b1, b2, b3 = st.columns(3)
        n_td = b1.number_input("Tekanan Darah", min_value=1, step=1, value=None, format="%d")
        n_rk = b2.selectbox("Riwayat Keluarga", ["0 - Tidak ada", "1 - Ada"])
        n_kls = b3.selectbox("Kelas Target", ["0 - Rendah", "1 - Tinggi"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        tambah = st.form_submit_button("Simpan Data ke Dataset", type="primary")
        
    if tambah:
        if not n_id.strip() or None in (n_usia, n_bmi, n_glu, n_td):
            st.error("❌ Mohon lengkapi seluruh kolom tambah data baru.")
        elif any(r[0] == n_id.strip() for r in data):
            st.error(f"❌ Kode pasien '{n_id.strip()}' sudah terdaftar dalam dataset.")
        else:
            data.append([n_id.strip(), int(n_usia), float(n_bmi), int(n_glu), int(n_td), int(n_rk[0]), int(n_kls[0])])
            simpan_data(data)
            st.success(f"✨ Data pasien {n_id.strip()} berhasil ditambahkan ke dataset!")
            st.rerun()

    st.markdown("---")
    st.subheader("Manajemen & Pembersihan Data")
    hapus = st.multiselect("Pilih kode pasien yang akan dihapus dari dataset", [r[0] for r in data])
    d1, d2 = st.columns(2)
    
    if d1.button("Hapus Data Terpilih") and hapus:
        if len(hapus) >= len(data):
            st.error("❌ Dataset tidak boleh kosong (minimal menyisakan 1 data).")
        else:
            st.session_state.data = [r for r in data if r[0] not in hapus]
            simpan_data(st.session_state.data)
            st.success("🗑️ Data yang dipilih berhasil dihapus.")
            st.rerun()
            
    if d2.button("Reset ke Dataset Awal Modul"):
        st.session_state.data = [r[:] for r in DEFAULT]
        simpan_data(st.session_state.data)
        st.success("🔄 Dataset berhasil dikembalikan ke pengaturan awal modul.")
        st.rerun()
