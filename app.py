"""
SPK Risiko Diabetes - K-Nearest Neighbors (Streamlit)
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

# ---------- tampilan ----------
st.set_page_config(page_title="SPK Risiko Diabetes - KNN", page_icon="🌸", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #fff5f8;
}
.hero{background:linear-gradient(120deg,#e6407f,#ff8fb8);color:white;padding:26px;border-radius:20px;
     text-align:center;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem;color:white}.hero p{margin:4px 0 0;color:white}
.hasil{border-radius:18px;padding:22px;text-align:center;color:white;margin-top:8px}
.hasil.t{background:linear-gradient(120deg,#d6264f,#ff6b8f)}
.hasil.r{background:linear-gradient(120deg,#2e9b6a,#6fd3a3)}
.hasil h2{color:white;margin:0 0 6px}

/* --- TOMBOL ESTETIK & LEMBUT --- */
.stButton>button, .stFormSubmitButton>button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #ff6b97, #e6407f) !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1.2rem !important;
    box-shadow: 0 4px 10px rgba(230, 64, 127, 0.25) !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    background: linear-gradient(135deg, #e6407f, #c7255f) !important;
    box-shadow: 0 6px 14px rgba(230, 64, 127, 0.4) !important;
    transform: translateY(-2px);
}
</style>
<div class="hero"><h1>🌸 Sistem Pendukung Keputusan Risiko Diabetes</h1>
<p>Klasifikasi pasien dengan algoritma K-Nearest Neighbors (KNN)</p></div>
""", unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = muat_data()
data = st.session_state.data

tab1, tab2 = st.tabs(["Prediksi", "Dataset"])

# ===== tab prediksi =====
with tab1:
    st.subheader("1. Input Data Pasien Baru")
    contoh = st.checkbox("Isi dengan contoh pasien baru dari soal (42, 28.5, 130, 128, 1)")
    with st.form("form_pasien"):
        c1, c2, c3, c4 = st.columns(4)
        nama = c1.text_input("Nama Pasien", value="Pasien Baru" if contoh else "")
        usia = c2.number_input("Usia (tahun)", min_value=1, step=1, value=42 if contoh else None, format="%d")
        bmi = c3.number_input("BMI", min_value=1.0, step=0.1, value=28.5 if contoh else None, format="%.1f")
        glu = c4.number_input("Glukosa (mg/dL)", min_value=1, step=1, value=130 if contoh else None, format="%d")
        
        c5, c6, c7 = st.columns(3)
        td = c5.number_input("Tekanan Darah (mmHg)", min_value=1, step=1, value=128 if contoh else None, format="%d")
        rk = c6.selectbox("Riwayat Diabetes Keluarga", ["0 - Tidak ada", "1 - Ada"],
                          index=1 if contoh else None, placeholder="Pilih")
        k = c7.number_input("Nilai K", min_value=1, max_value=len(data), step=1, value=min(3, len(data)), format="%d")
        kirim = st.form_submit_button("Hitung KNN", type="primary")

    if kirim:
        if not nama.strip() or rk is None or None in (usia, bmi, glu, td):
            st.error("Semua kolom wajib diisi: nama, usia, BMI, glukosa, tekanan darah, dan riwayat keluarga.")
        else:
            x = [float(usia), float(bmi), float(glu), float(td), int(rk[0])]
            h = hitung_knn(data, x, int(k))
            k = int(k)

            st.subheader("2. Proses Perhitungan")
            st.markdown("**Langkah 1 - Data pasien**")
            df_pasien = pd.DataFrame([[nama] + x], columns=["Nama"] + FITUR)
            df_pasien["BMI"] = df_pasien["BMI"].apply(lambda val: f"{val:.1f}")
            st.dataframe(df_pasien, hide_index=True, use_container_width=True)

            st.markdown("**Langkah 2 - Normalisasi Min-Max**")
            st.latex(r"x' = \frac{x - \min}{\max - \min}")
            st.dataframe(pd.DataFrame({"Atribut": FITUR, "Min": h["mn"], "Max": h["mx"],
                                       "Nilai pasien": x, "Hasil normalisasi": [round(v, 4) for v in h["xn"]]}),
                         hide_index=True, use_container_width=True)

            st.markdown("**Langkah 3 - Jarak Euclidean ke setiap data latih**")
            st.latex(r"d = \sqrt{\sum (x'_{pasien} - x'_{latih})^2}")
            r0 = h["rows"][0]
            suku = " + ".join(f"({a:.3f}-{b:.3f})^2" for a, b in zip(h["xn"], r0["fn"]))
            st.caption(f"Contoh {r0['id']}: d = akar({suku}) = {r0['d']:.4f}")
            
            df_jarak = pd.DataFrame([[i + 1, r["id"]] + r["f"] + [round(r["d"], 4), label(r["kelas"])]
                                     for i, r in enumerate(h["rows"])],
                                    columns=["Peringkat", "Pasien"] + FITUR + ["Jarak", "Kelas"])
            df_jarak["BMI"] = df_jarak["BMI"].apply(lambda val: f"{val:.1f}")
            
            warna = lambda row: ["background-color:#ffb3d1;font-weight:bold" if row["Peringkat"] <= k else ""
                                 for _ in row]
            st.dataframe(df_jarak.style.apply(warna, axis=1), hide_index=True, use_container_width=True)
            st.caption(f"Baris berwarna adalah {k} tetangga terdekat.")

            st.markdown(f"**Langkah 4 - Voting K = {k} tetangga terdekat**")
            st.write("Tetangga: " + ", ".join(f"{r['id']} ({label(r['kelas'])})" for r in h["nb"]))
            m1, m2 = st.columns(2)
            m1.metric("Suara Risiko Tinggi", h["tinggi"])
            m2.metric("Suara Risiko Rendah", h["rendah"])
            if h["tinggi"] == h["rendah"]:
                st.caption("Suara seimbang, kelas ditentukan dari tetangga terdekat.")

            st.subheader("3. Kesimpulan")
            t = h["hasil"] == 1
            st.markdown(
                f"""<div class="hasil {'t' if t else 'r'}">
                <h2>{'⚠️ RISIKO DIABETES TINGGI' if t else '✅ RISIKO DIABETES RENDAH'}</h2>
                Berdasarkan {k} tetangga terdekat ({h['tinggi']} Tinggi, {h['rendah']} Rendah),
                pasien <b>{nama}</b> ({int(usia)} tahun) diklasifikasikan sebagai
                <b>{'Risiko Tinggi (kelas 1)' if t else 'Risiko Rendah (kelas 0)'}</b>.<br>
                {'Disarankan pemeriksaan lanjutan dan konsultasi dokter.' if t else
                 'Pertahankan pola hidup sehat dan lakukan pemeriksaan berkala.'}</div>""",
                unsafe_allow_html=True)

# ===== tab dataset =====
with tab2:
    st.subheader("Dataset Latih")
    df_latih = pd.DataFrame([r[:6] + [label(r[6])] for r in data],
                            columns=["Pasien"] + FITUR + ["Kelas"])
    df_latih["BMI"] = df_latih["BMI"].apply(lambda val: f"{val:.1f}")
    st.dataframe(df_latih, hide_index=True, use_container_width=True)

    st.subheader("Tambah Data Baru ke Dataset")
    with st.form("form_tambah", clear_on_submit=True):
        a1, a2, a3, a4 = st.columns(4)
        n_id = a1.text_input("Kode Pasien")
        n_usia = a2.number_input("Usia", min_value=1, step=1, value=None, format="%d")
        n_bmi = a3.number_input("BMI", min_value=1.0, step=0.1, value=None, format="%.1f")
        n_glu = a4.number_input("Glukosa", min_value=1, step=1, value=None, format="%d")
        b1, b2, b3 = st.columns(3)
        n_td = b1.number_input("Tekanan Darah", min_value=1, step=1, value=None, format="%d")
        n_rk = b2.selectbox("Riwayat Keluarga", ["0 - Tidak ada", "1 - Ada"])
        n_kls = b3.selectbox("Kelas", ["0 - Rendah", "1 - Tinggi"])
        tambah = st.form_submit_button("Tambah ke Dataset", type="primary")
    if tambah:
        if not n_id.strip() or None in (n_usia, n_bmi, n_glu, n_td):
            st.error("Lengkapi semua kolom data baru.")
        elif any(r[0] == n_id.strip() for r in data):
            st.error("Kode pasien sudah ada, gunakan kode lain.")
        else:
            data.append([n_id.strip(), int(n_usia), float(n_bmi), int(n_glu), int(n_td), int(n_rk[0]), int(n_kls[0])])
            simpan_data(data)
            st.success(f"Data {n_id.strip()} berhasil ditambahkan.")
            st.rerun()

    st.subheader("Hapus Data")
    hapus = st.multiselect("Pilih kode pasien yang akan dihapus", [r[0] for r in data])
    d1, d2 = st.columns(2)
    if d1.button("Hapus terpilih") and hapus:
        if len(hapus) >= len(data):
            st.error("Dataset minimal berisi 1 data.")
        else:
            st.session_state.data = [r for r in data if r[0] not in hapus]
            simpan_data(st.session_state.data)
            st.rerun()
    if d2.button("Kembalikan dataset awal"):
        st.session_state.data = [r[:] for r in DEFAULT]
        simpan_data(st.session_state.data)
        st.rerun()
