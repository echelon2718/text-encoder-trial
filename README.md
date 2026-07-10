# T-LeJEPA: Text-LeJEPA untuk Representasi Kanonik-Invarian pada Teks Non-Standar

## Metodologi Penelitian

---

## 1. Pendahuluan dan Motivasi

Bahasa tulis informal — pesan singkat, media sosial, atau percakapan sehari-hari — memuat variasi leksikal yang sangat besar untuk menyatakan makna yang secara esensial identik. Ekspresi *"idk"*, *"i dnt knw"*, *"idontknow"*, dan *"i dunno"*, misalnya, secara sintaktik berbeda jauh dari bentuk baku *"I don't know"*, namun secara semantik dipahami penutur sebagai hal yang sama. Fenomena ini telah lama menjadi perhatian dalam riset *lexical normalization*: Han & Baldwin (2011) dan Han, Cook, & Baldwin (2013) menunjukkan bahwa kata-kata *out-of-vocabulary* (OOV) pada teks mikroblog dapat dipetakan ke bentuk standarnya melalui kombinasi kemiripan morfofonemik dan konteks kalimat, tanpa memerlukan anotasi berlabel. Namun, pendekatan normalisasi klasik umumnya bekerja pada level kata secara diskrit (candidate generation–selection) dan tidak menghasilkan *representasi vektor* yang dapat langsung dipakai untuk tugas hilir (*downstream*) seperti pencarian semantik atau *clustering*.

Di sisi lain, perkembangan *self-supervised representation learning*, khususnya *Joint-Embedding Predictive Architecture* (JEPA) yang diperkenalkan LeCun (2022) dan diinstansiasi pertama kali pada citra melalui I-JEPA (Assran et al., 2023), menawarkan paradigma yang menarik: alih-alih merekonstruksi input secara piksel demi piksel, model dilatih untuk menyamakan *representasi laten* dari berbagai *view* suatu data. Perkembangan terakhir, **LeJEPA** (Balestriero & LeCun, 2025), memberikan fondasi teoretis yang kokoh untuk JEPA melalui regularisasi **SIGReg** (*Sketched Isotropic Gaussian Regularization*), yang membuktikan bahwa distribusi *Gaussian isotropik* adalah distribusi optimal bagi *embedding* untuk meminimalkan risiko prediksi pada tugas hilir yang belum diketahui, sekaligus mencegah *representation collapse* tanpa heuristik seperti *stop-gradient* atau arsitektur *teacher-student*.

Pertanyaan yang diajukan pada pekerjaan ini adalah: **dapatkah kerangka LeJEPA diadaptasi untuk domain tekstual, di mana setiap titik data memiliki satu bentuk kanonik (baku) dan sejumlah bentuk non-kanonik (variasi informal), sedemikian rupa sehingga seluruh variasi tersebut dipetakan ke representasi laten yang identik dengan bentuk kanoniknya?** Kami berargumen bahwa adaptasi langsung LeJEPA ke ranah teks tidak memadai, karena bahasa —tidak seperti citra— memiliki dua lapisan makna yang harus ditangani secara eksplisit dan terpisah: **sintaktik** (kesesuaian struktur simbol dengan tata bahasa) dan **semantik** (makna yang disampaikan struktur tersebut). Kami mengusulkan **T-LeJEPA** (*Text-LeJEPA*), yang memperluas *loss* LeJEPA dengan dekomposisi eksplisit menjadi komponen sintaktik dan semantik, ditambah mekanisme prediksi panjang kanonik yang terinspirasi dari literatur *non-autoregressive machine translation* (Gu et al., 2018).

Sebagai catatan metodologis, bagian ini menyajikan **rumusan teoretis dan justifikasi desain** dari T-LeJEPA berdasarkan argumen matematis dan dukungan literatur; validasi empiris (kurva pelatihan, evaluasi *downstream*, ablasi komponen) berada di luar cakupan bagian ini dan didiskusikan sebagai agenda kerja pada Bagian 7.

---

## 2. Rumusan Masalah

### 2.1 Definisi Formal Dataset

Diberikan korpus $\mathcal{D} = \{X^{(i)}\}_{i=1}^{N}$, di mana setiap titik data merupakan himpunan *multi-view*:

$$
X^{(i)} = \{x_1^{(i)}, x_2^{(i)}, \ldots, x_V^{(i)}\}, \qquad i \in \{1, \ldots, N\}
$$

dengan $x_1^{(i)}$ adalah bentuk **kanonik** (baku, gramatikal) dari kalimat ke-$i$, dan $\{x_2^{(i)}, \ldots, x_V^{(i)}\}$ adalah $V-1$ bentuk **non-kanonik** (variasi informal, disingkat, atau bernoise) dari kalimat yang sama. Setiap $x_j^{(i)}$ adalah barisan indeks token dengan panjang $L^{(i,j)}$ yang berbeda-beda antar view dan antar data. Untuk keperluan implementasi berbasis *mini-batch* berukuran $B$, satu batch data direpresentasikan sebagai tensor $x_j^{(i)} \in \mathbb{R}^{B \times L^{(i,j)}}$ dengan *padding* menyesuaikan panjang maksimum dalam batch; secara konseptual, pembahasan berikut mendeskripsikan operasi per satu titik data $i$.

**Tujuan.** Membangun sebuah *embedding network* $f_\theta$ sedemikian rupa sehingga untuk setiap $i$ dan setiap $v \in \{1, \ldots, V\}$:

$$
f_\theta(x_v^{(i)}) \approx f_\theta(x_1^{(i)})
$$

Dengan kata lain, seluruh variasi permukaan (*surface form*) dari suatu kalimat harus konvergen ke satu titik laten yang sama dengan bentuk kanoniknya — sebuah sifat yang dalam literatur representasi kalimat disebut *invariance* terhadap variasi non-semantik (bandingkan dengan tujuan *augmentation-invariance* pada SimCLR, Chen et al., 2020, dan pada VICReg, Bardes, Ponce, & LeCun, 2022, untuk domain citra).

### 2.2 Sintaktik vs. Semantik: Dua Sumbu Pemahaman Bahasa

Dalam linguistik dan pemrosesan bahasa alami, **sintaktik** mengacu pada kesesuaian susunan simbol terhadap kaidah gramatikal suatu bahasa, sedangkan **semantik** mengacu pada makna yang terkandung dalam susunan tersebut. Kedua sumbu ini tidak selalu berkorelasi: sebuah kalimat dapat memenuhi kaidah sintaktik namun tidak bermakna (*"the bone eats the dog"* adalah kalimat gramatikal namun secara semantik anomali), sebaliknya, bentuk seperti *"idk"* atau *"i dnt knw"* secara sintaktik menyimpang jauh dari tata baku, tetapi tetap dapat dipahami maknanya oleh penutur — sebuah fenomena yang menjadi dasar seluruh riset *lexical normalization* (Han & Baldwin, 2011; Han, Cook, & Baldwin, 2013). Yang menarik, derajat pemahaman terhadap bentuk non-standar ini bersifat kontekstual dan bergantung pada latar belakang sosiolinguistik pembaca/penutur, sehingga *ground truth* semantik untuk bentuk non-kanonik tidak selalu bersifat objektif tunggal.

Implikasi bagi desain model: sebuah fungsi *loss* tunggal yang hanya menyamakan titik-titik laten (seperti $\mathcal{L}_{\text{align}}$ pada LeJEPA orisinal) secara implisit hanya menegakkan invariansi struktural/sintaktik antar-view, tetapi tidak menjamin bahwa jarak antar-representasi di ruang laten mencerminkan **derajat kemiripan makna** antar kalimat yang berbeda. Dengan demikian, T-LeJEPA memerlukan komponen *loss* terpisah yang secara eksplisit menegakkan struktur semantik pada ruang representasi — inilah motivasi $\mathcal{L}_{\text{semantic}}$ pada Bagian 4.

### 2.3 Ketidakpastian Epistemik vs. Aleatorik

Dipinjam dari literatur kuantifikasi ketidakpastian pada pembelajaran mendalam (Kendall & Gal, 2017), kami membedakan dua sumber ketidakpastian yang relevan bagi desain *loss*:

- **Ketidakpastian epistemik**, yakni ketidakpastian akibat kekurangan data atau pengetahuan model. Dalam konteks *self-supervised learning*, tidak terdapat *ground truth* eksternal (label manusia); "label" yang tersedia hanyalah representasi vektor yang dihasilkan model itu sendiri. Karena itu, ketidakpastian epistemik pada dasarnya dapat direduksi dengan memperbesar cakupan korpus pelatihan (lebih banyak variasi non-kanonik teramati per kalimat kanonik).
- **Ketidakpastian aleatorik**, yakni ketidakpastian yang melekat pada data akibat data tersebut memiliki lebih dari satu nilai kebenaran yang valid secara bersamaan. Contoh klasik adalah polisemi leksikal — kata *"bank"* dapat bermakna institusi finansial atau tepian sungai (*"river bank"*) — sebuah bentuk ambiguitas yang telah dipetakan secara ekstensif dalam literatur *word sense disambiguation* (Navigli, 2009). Ketidakpastian jenis ini tidak dapat direduksi hanya dengan menambah data, karena tidak ada sinyal supervisi yang secara eksplisit memberi tahu model bagian mana dari data yang bersifat ambigu.

Dua pendekatan lazim untuk menangani ketidakpastian aleatorik adalah: (i) memberi label eksplisit pada tingkat ambiguitas suatu kalimat (mahal dan tidak trivial untuk diskalakan), atau (ii) membiarkan model mempelajarinya secara implisit lewat konteks kalimat yang lebih luas — pendekatan yang diambil oleh model bahasa kontekstual seperti BERT (Devlin et al., 2019) dan yang juga kami adopsi di T-LeJEPA, melalui *encoder* dua-arah (*bidirectional*) yang mengkondisikan representasi setiap token pada seluruh konteks kalimat. Implikasinya, T-LeJEPA dirancang sebagai model **deterministik** (bukan probabilistik): tanpa sinyal supervisi eksplisit tentang mana data yang aleatorik, memaksakan keluaran multi-modal/probabilistik hanya akan menambah derajat kebebasan model tanpa mekanisme pelatihan yang valid untuk mengarahkannya. Penanganan ambiguitas aleatorik dilakukan secara implisit melalui kapasitas *encoder* bidireksional dalam menangkap fitur kontekstual N-gram, bukan lewat pemodelan distribusi eksplisit atas makna.

---

## 3. Karya Terkait

### 3.1 Joint-Embedding Predictive Architecture dan Pencegahan Collapse

JEPA (LeCun, 2022) memformulasikan pembelajaran representasi sebagai prediksi dalam ruang laten, bukan ruang piksel/token, dengan tujuan menghindari alokasi kapasitas model pada detail berlevel rendah yang tidak relevan secara semantik. I-JEPA (Assran et al., 2023) menjadi instansiasi praktis pertama pada citra, menggunakan arsitektur *context–target* dengan *predictor network*, *target encoder* berbasis *exponential moving average* (EMA), dan *stop-gradient* untuk mencegah *collapse* — yakni kondisi degeneratif di mana model memetakan seluruh input ke satu titik konstan, sehingga *loss* menjadi nol namun representasi tidak informatif.

Untuk mengatasi *collapse* tanpa heuristik semacam itu, dua garis kerja regularisasi menjadi rujukan penting: **VICReg** (Bardes, Ponce, & LeCun, 2022) menambahkan suku varians dan kovarians eksplisit agar setiap dimensi *embedding* mempertahankan variansi minimum dan saling tidak berkorelasi; **Barlow Twins** (Zbontar, Jing, Misra, LeCun, & Deny, 2021) mendekati masalah yang sama lewat matriks korelasi silang antar dua *view*, mendorong matriks tersebut mendekati matriks identitas (redundancy reduction). **LeJEPA** (Balestriero & LeCun, 2025) menggeneralisasi gagasan ini secara teoretis: alih-alih menegakkan varians/kovarians tertentu secara ad hoc, LeJEPA membuktikan bahwa distribusi target optimal bagi *embedding* — dalam pengertian meminimalkan *risk* rata-rata di atas seluruh kemungkinan tugas hilir *linear* dan *non-linear* — adalah **distribusi Gaussian isotropik** $\mathcal{N}(\mathbf{0}, \mathbf{I})$. Regularizer yang diusulkan, **SIGReg**, menegakkan hal ini dengan mencocokkan fungsi karakteristik empiris dari proyeksi 1-dimensi $\{a^\top z_n\}_{n=1}^N$ (untuk sejumlah arah proyeksi acak $a \in A$) terhadap fungsi karakteristik Gaussian standar, menggunakan uji normalitas berbasis fungsi karakteristik seperti uji Epps–Pulley. Validitas pendekatan berbasis proyeksi 1-dimensi ini dijamin oleh teorema Cramér–Wold, yang menyatakan bahwa distribusi multivariat sepenuhnya ditentukan oleh seluruh proyeksi 1-dimensinya. Keunggulan praktis SIGReg adalah kompleksitas waktu dan memori yang **linear**, serta terbukti stabil pada lebih dari 60 arsitektur berbeda tanpa penyetelan hiperparameter spesifik-arsitektur.

T-LeJEPA meminjam mesin SIGReg ini secara langsung untuk mencegah *collapse* pada level *encoder* teks, namun — sebagaimana akan dijelaskan pada Bagian 4 dan 5 — menempatkannya pada posisi arsitektural yang berbeda dari LeJEPA orisinal, karena adanya kebutuhan tambahan berupa *decoder* kanonik yang tidak ada pada JEPA untuk citra.

### 3.2 Representasi Kalimat, Normalisasi Leksikal, dan Distilasi Kemiripan

Pada sisi representasi kalimat, **SimCSE** (Gao, Yao, & Chen, 2021) menunjukkan bahwa *dropout* sebagai augmentasi minimal, dikombinasikan dengan *loss* kontrastif berbasis InfoNCE, menghasilkan *embedding* kalimat yang sangat kompetitif pada tugas *Semantic Textual Similarity* (STS), sekaligus membuktikan secara teoretis dan empiris bahwa *loss* kontrastif meregularisasi ruang *embedding* pra-latih yang anisotropik menjadi lebih seragam (*uniform*). **Sentence-BERT** (Reimers & Gurevych, 2019) memperkenalkan arsitektur *siamese*/*triplet* di atas BERT dengan strategi *pooling* (rata-rata token, `[CLS]`, atau maksimum) untuk menghasilkan vektor kalimat berdimensi tetap yang dapat dibandingkan langsung via kesamaan kosinus — strategi *pooling* yang menjadi dasar operator $\mu_L(\cdot)$ pada rumusan $\mathcal{L}_{\text{semantic}}$ kami (Bagian 4.3).

Pada sisi normalisasi teks non-standar, garis kerja Han & Baldwin (2011, 2013) menunjukkan bahwa variasi leksikal informal dapat dipetakan ke bentuk baku melalui klasifikasi kandidat berbasis kemiripan morfofonemik dan konteks, tanpa anotasi. T-LeJEPA dapat dipandang sebagai generalisasi tujuan ini dari **level kata diskrit** ke **level kalimat kontinu**: alih-alih memetakan setiap kata OOV ke padanan kamusnya, kami memetakan seluruh kalimat non-kanonik ke satu titik pada ruang *embedding* berkelanjutan yang berimpit dengan representasi kalimat kanoniknya.

Terakhir, mekanisme $\mathcal{L}_{\text{semantic}}$ yang mencocokkan **matriks kemiripan berpasangan** (bukan titik individual) antara representasi model dan skor kemiripan dari model guru (*teacher*) berakar dari literatur distilasi relasional: **Relational Knowledge Distillation** (Park, Kim, Lu, & Cho, 2019) memperkenalkan *loss* berbasis jarak dan sudut antar-contoh alih-alih hanya mencocokkan aktivasi tiap contoh secara independen; **Similarity-Preserving Knowledge Distillation** (Tung & Mori, 2019) secara lebih spesifik mencocokkan matriks kemiripan berpasangan (mirip matriks Gram) antara model siswa dan guru. Prinsip yang sama kami adopsi: model guru semantik — dapat berupa SimCSE atau Sentence-BERT yang telah dilatih pada data NLI/STS — menyediakan skor kemiripan $s_{\text{teacher}}(x_{i,1}, x_{j,1})$ sebagai sinyal distilasi lunak (*soft label*) untuk pasangan kalimat kanonik, alih-alih label biner keras. Pendekatan distilasi lunak semacam ini konsisten dengan kerangka distilasi pengetahuan klasik (Hinton, Vinyals, & Dean, 2015) dan secara langsung menjawab tantangan ketidakpastian aleatorik pada Bagian 2.3: karena kemiripan makna antar kalimat sering bersifat gradual/kontinu (bukan biner), skor lunak dari model guru lebih sesuai daripada label pasangan biner.

### 3.3 Prediksi Panjang pada Generasi Sekuens Non-Autoregresif

Karena *decoder* kanonik T-LeJEPA perlu menghasilkan keluaran dengan panjang tertentu yang *diketahui di muka* (bukan dihasilkan token demi token secara autoregresif), kami mengadopsi strategi prediksi panjang dari literatur *Non-Autoregressive Neural Machine Translation* (NAT). Gu et al. (2018) memperkenalkan prediktor "fertilitas"/panjang yang mengambil representasi *encoder* sebagai masukan dan memprediksi panjang sekuens target; selama pelatihan, panjang **sebenarnya** (bukan hasil prediksi) yang digunakan untuk membentuk kanvas keluaran *decoder* — sebuah strategi yang dikenal sebagai *teacher forcing* — sementara prediktor panjang tetap dilatih secara terpisah terhadap panjang target sebenarnya. Pada saat inferensi, barulah panjang hasil prediksi dipakai untuk menentukan ukuran kanvas *decoder*. Skema inilah yang kami adaptasi untuk komponen $\tau_\psi$ pada T-LeJEPA (Bagian 4.5).

---

## 4. Metodologi yang Diusulkan: T-LeJEPA

### 4.1 Komponen Model dan Alur Propagasi Maju

T-LeJEPA melibatkan tiga jaringan terparametrisasi:

| Simbol | Peran | Analog Arsitektural |
|---|---|---|
| $f_\theta$ | *Text encoder* (Transformer *encoder*, bidireksional) | Peta $x \mapsto z$ |
| $g_\phi$ | *Canonical decoder* (Transformer *decoder*) | Peta $z \mapsto z_c$, mengembalikan panjang kanonik |
| $\tau_\psi$ | *Length predictor* (Feed-Forward Network) | Peta $z \mapsto \hat{L}$ |

Alur propagasi maju untuk titik data ke-$i$ adalah sebagai berikut. Pertama, seluruh view $\{x_1^{(i)}, \ldots, x_V^{(i)}\}$ diubah menjadi *embedding* token melalui lapisan *embedding* (termasuk *tokenization* subword bila relevan):

$$
X_{\text{emb}}^{(i)} = \{x_{\text{emb},1}^{(i)}, \ldots, x_{\text{emb},V}^{(i)}\}, \qquad x_{\text{emb},j}^{(i)} \in \mathbb{R}^{B \times C^{(i,j)} \times d}
$$

dengan $C^{(i,j)}$ adalah panjang sekuens setelah tokenisasi (dapat berbeda dari $L^{(i,j)}$ akibat *subword splitting*) dan $d$ adalah dimensi *embedding*. Setiap view diteruskan ke *encoder* bersama (*weight-sharing* antar-view, sebagaimana lazim pada arsitektur *siamese* JEPA):

$$
f_\theta(X_{\text{emb}}^{(i)}) = Z^{(i)} = \{z_1^{(i)}, \ldots, z_V^{(i)}\}
$$

Representasi $Z^{(i)}$ kemudian diteruskan secara paralel ke *canonical decoder* dan *length predictor*:

$$
g_\phi(Z^{(i)}) = Z_c^{(i)} = \{z_{c,1}^{(i)}, \ldots, z_{c,V}^{(i)}\}, \qquad \tau_\psi(Z^{(i)}) \to \hat{L}^{(i)}
$$

**Batasan panjang keluaran.** Karena tujuan *decoder* adalah menerjemahkan setiap view non-kanonik ke bentuk kanonik, keluaran $z_{c,1}^{(i)}, \ldots, z_{c,V}^{(i)}$ dipaksa memiliki panjang sekuens yang sama dengan $z_1^{(i)}$ (panjang kanonik $L_{i,1}$). Selama pelatihan, panjang ini diberikan langsung dari data (*teacher forcing*, lihat Bagian 4.5); *decoder* tidak bergantung pada keluaran $\tau_\psi$ untuk menentukan ukuran kanvasnya sendiri saat pelatihan.

### 4.2 Loss Sintaktik: Penegakan Invariansi Struktural

Komponen pertama menegakkan bahwa representasi hasil *decoder* dari setiap view — setelah diterjemahkan ke ruang kanonik — sama dengan representasi *encoder* dari bentuk kanonik aslinya:

$$
\mathcal{L}_{\text{syntactic}} = \frac{1}{NV} \sum_{n=1}^{N} \sum_{v=1}^{V} \left\| z_{c,n,v} - z_{n,1} \right\|_2^2 = \frac{1}{NV} \sum_{n=1}^{N} \sum_{v=1}^{V} \left\| g_\phi\big(f_\theta(x_{n,v}),\, L_{n,1}\big) - f_\theta(x_1^{(n)}) \right\|_2^2
$$

Perhatikan bahwa **target** dari *loss* ini, $z_{n,1} = f_\theta(x_1^{(n)})$, adalah keluaran *encoder* atas teks kanonik itu sendiri — bukan rata-rata seluruh view seperti pada suku $\mathcal{L}_{\text{align}}$ LeJEPA orisinal ($\|z_{n,v} - \mu_n\|_2^2$ dengan $\mu_n = \frac{1}{V}\sum_v z_{n,v}$). Perbedaan ini bukan sekadar variasi teknis, melainkan konsekuensi logis dari perbedaan struktur data: pada JEPA citra, seluruh *view* (hasil augmentasi acak) berkedudukan setara — tidak ada "citra kanonik" — sehingga rata-rata populasi adalah estimator yang wajar bagi konten invarian. Pada T-LeJEPA, sebaliknya, terdapat *anchor* yang secara eksplisit diberikan oleh data ($x_1^{(i)}$), sehingga menggunakan rata-rata alih-alih anchor sebenarnya hanya akan menambah varians estimasi tanpa manfaat, sekaligus mengabaikan informasi yang tersedia secara *supervised* implisit dari struktur dataset. Justifikasi ini konsisten dengan prinsip umum estimasi statistik: ketika target sebenarnya tersedia, menggantinya dengan estimator (rata-rata sampel) hanya rasional jika target sebenarnya tidak teramati.

### 4.3 Loss Semantik: Distilasi Struktur Kemiripan Berpasangan

Komponen kedua menjawab keterbatasan $\mathcal{L}_{\text{syntactic}}$ yang disebutkan pada Bagian 2.2: penyamaan titik laten semata tidak menjamin bahwa *geometri* ruang laten antar kalimat yang berbeda mencerminkan kemiripan maknanya. Untuk itu, kami mendefinisikan operator *pooling* $\mu_L(\cdot)$ yang memetakan representasi token-level $z_{c,i,v} \in \mathbb{R}^{C \times d}$ menjadi vektor kalimat berdimensi tetap $\mu_L(z_{c,i,v}) \in \mathbb{R}^{d}$ melalui rata-rata sepanjang sumbu sekuens — strategi *mean pooling* yang mengikuti Sentence-BERT (Reimers & Gurevych, 2019):

$$
\mu_L(z) = \frac{1}{L} \sum_{t=1}^{L} z_t
$$

Untuk setiap pasangan data $(i,j)$ dan setiap pasangan view $(v_1, v_2)$, kami mendefinisikan residual antara kemiripan kosinus hasil model dan skor guru semantik eksternal $s_{\text{teacher}}$ (mis. dari SimCSE atau Sentence-BERT terlatih pada data STS/NLI, dievaluasi pada pasangan kalimat kanonik $x_{i,1}, x_{j,1}$):

$$
d_{ij}^{(v_1,v_2)} = \operatorname{sim}\big(\mu_L(z_{c,i,v_1}),\, \mu_L(z_{c,j,v_2})\big) - s_{\text{teacher}}(x_{i,1}, x_{j,1})
$$

$$
D_{ij} = \begin{bmatrix} d_{ij}^{(1,1)} & \cdots & d_{ij}^{(1,V)} \\ \vdots & \ddots & \vdots \\ d_{ij}^{(V,1)} & \cdots & d_{ij}^{(V,V)} \end{bmatrix}, \qquad \delta_{ij} = \sum_{v_1=1}^{V}\sum_{v_2=1}^{V} \left(d_{ij}^{(v_1,v_2)}\right)^2 = \|D_{ij}\|_F^2
$$

$$
\mathcal{L}_{\text{semantic}} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \delta_{ij} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \|D_{ij}\|_F^2
$$

Rumusan ini secara struktural setara dengan **distilasi matriks kemiripan berpasangan** (*similarity-preserving distillation*): alih-alih mencocokkan aktivasi tiap contoh secara independen terhadap model guru, kami mencocokkan **seluruh matriks kemiripan lintas-view lintas-data** terhadap struktur kemiripan yang diberikan model guru, mengikuti prinsip yang diperkenalkan Tung & Mori (2019) dan diperluas Park et al. (2019) dalam bentuk *distance-wise*/*angle-wise loss*. Peran model guru $s_{\text{teacher}}$ di sini adalah menyediakan sinyal semantik lunak (*soft, continuous label*) — bukan label biner keras — yang sesuai dengan sifat gradual kemiripan makna, sekaligus menjadi jawaban langsung terhadap tantangan ketidakpastian aleatorik yang dibahas pada Bagian 2.3: karena tidak ada anotasi manusia yang secara eksplisit menandai derajat ambiguitas suatu kalimat, sinyal distilasi dari model guru semantik pra-latih menjadi proksi yang dapat diskalakan tanpa anotasi tambahan.

### 4.4 SIGReg: Pencegahan *Representation Collapse*

Mengikuti Balestriero & LeCun (2025), kami menerapkan SIGReg pada **keluaran mentah encoder** $f_\theta(x_{\text{emb}}^{(n)})$ — bukan pada keluaran *decoder* $Z_c$:

$$
\mathcal{L}_{\text{SIGReg}} = \frac{1}{V|A|} \sum_{v=1}^{V} \sum_{a \in A} T\Big(\big\{a^\top f_\theta(x_{\text{emb}}^{(n)})\big\}_{n=1}^{N}\Big)
$$

dengan $A$ adalah himpunan arah proyeksi acak dan $T(\cdot)$ adalah statistik uji normalitas berbasis fungsi karakteristik (mis. Epps–Pulley), yang mengukur seberapa jauh distribusi proyeksi 1-dimensi $\{a^\top z_n\}_n$ menyimpang dari distribusi Gaussian standar $\mathcal{N}(0,1)$. Validitas pendekatan berbasis proyeksi ini dijamin teorema Cramér–Wold: kesesuaian seluruh proyeksi 1-dimensi terhadap Gaussian standar setara dengan kesesuaian distribusi multivariat penuh terhadap $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Justifikasi penempatan pada ruang encoder, bukan ruang decoder.** Terdapat dua alasan. *Pertama*, secara praktis, $\mathcal{L}_{\text{SIGReg}}$ perlu diterapkan pada representasi yang dibagikan (*shared*) oleh seluruh view — kanonik maupun non-kanonik — agar seluruh cabang input tunduk pada batasan non-degenerasi yang sama; keluaran *encoder* $Z^{(i)}$ memenuhi syarat ini secara alami karena seluruh view melewatinya. *Kedua*, dan lebih penting secara konseptual: jika Gaussianitas isotropik dipaksakan pada ruang **keluaran decoder** $Z_c$, hal ini berpotensi bertentangan dengan tujuan $\mathcal{L}_{\text{syntactic}}$ dan $\mathcal{L}_{\text{semantic}}$, yang keduanya menuntut struktur geometris tertentu pada $Z_c$ (kedekatan ke anchor kanonik, dan pelestarian struktur kemiripan semantik) — dua tuntutan yang tidak niscaya kompatibel dengan bentuk distribusi Gaussian isotropik generik. Dengan menempatkan SIGReg pada $Z$ (ruang *encoder*) dan membiarkan $g_\phi$ berperan murni sebagai *translator* kanonik, kedua tujuan tidak saling berkompetisi: pencegahan *collapse* ditegakkan pada representasi bersama sebelum spesialisasi kanonik terjadi, sementara *decoder* bebas mempelajari pemetaan ke ruang kanonik tanpa batasan distribusi tambahan. Sebagai konsekuensi logisnya, apabila $g_\phi$ berhasil mempelajari pemetaan yang akurat dari bentuk non-kanonik ke ruang kanonik (via $\mathcal{L}_{\text{syntactic}}$), maka ruang kanonik yang dipelajari itu — karena berasal dari $z_{n,1} = f_\theta(x_1^{(n)})$, yang juga tunduk pada SIGReg — **mewarisi** sifat non-degeneratifnya secara tidak langsung, tanpa perlu regularisasi ganda.

Kami juga mencatat bahwa $\mathcal{L}_{\text{semantic}}$ tidak berkompetisi dengan $\mathcal{L}_{\text{SIGReg}}$ karena keduanya beroperasi pada ruang representasi yang berbeda secara fungsional: $\mathcal{L}_{\text{SIGReg}}$ menegakkan bentuk distribusi marginal pada ruang *encoder* $Z$ (per-dimensi, lintas-sampel), sedangkan $\mathcal{L}_{\text{semantic}}$ menegakkan struktur relasional berpasangan pada ruang *pooled* hasil *decoder* $\mu_L(Z_c)$ (antar-sampel, dalam bentuk matriks kemiripan) — keduanya adalah batasan pada aspek statistik yang berbeda dari representasi (bentuk distribusi vs. struktur relasi), sehingga secara aljabar tidak saling meniadakan.

### 4.5 Prediktor Panjang Kanonik dan *Teacher Forcing*

Mengadopsi skema Gu et al. (2018), $\tau_\psi$ dilatih meregresi panjang kanonik sebenarnya dari representasi *encoder* setiap view:

$$
\mathcal{L}_{\text{canonlen}} = \frac{1}{NV} \sum_{n=1}^{N} \sum_{v=1}^{V} \left\| L_{n,1} - \hat{L}_{n,v} \right\|_2^2 = \frac{1}{NV} \sum_{n=1}^{N} \sum_{v=1}^{V} \left\| L_{n,1} - \tau_\psi(z_{n,v}) \right\|_2^2
$$

Selama pelatihan, panjang kanvas keluaran $g_\phi$ ditentukan oleh panjang kanonik **sebenarnya** $L_{n,1}$ (teacher forcing), bukan oleh $\hat{L}_{n,v}$ — memastikan sinyal gradien pada $\mathcal{L}_{\text{syntactic}}$ dan $\mathcal{L}_{\text{semantic}}$ tidak terganggu oleh kesalahan prediksi panjang pada tahap awal pelatihan, ketika $\tau_\psi$ belum akurat. $\tau_\psi$ tetap dilatih secara paralel terhadap panjang sebenarnya, dan **hanya digunakan saat inferensi**, ketika $L_{n,1}$ tidak lagi tersedia untuk view non-kanonik yang belum diketahui bentuk bakunya.

### 4.6 Objektif Latihan Total

Secara konseptual, tiga komponen di atas dapat dikelompokkan ke dalam dua tujuan utama — pembentukan *embedding* kanonik (sintaktik + semantik) dan pencegahan *collapse* (SIGReg):

$$
\mathcal{L}_{\text{T-LeJEPA}} = \underbrace{\underbrace{\frac{1}{NV}\sum_{n=1}^{N}\sum_{v=1}^{V} \|z_{c,n,v} - z_{n,1}\|_2^2}_{\mathcal{L}_{\text{syntactic}}} + \underbrace{\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N} \|D_{ij}\|_F^2}_{\mathcal{L}_{\text{semantic}}}}_{\text{Canonical Embedding Objective}} \;+\; \underbrace{\frac{1}{V|A|}\sum_{v=1}^{V}\sum_{a \in A} T\big(\{a^\top f_\theta(x_{\text{emb}}^{(n)})\}_{n=1}^{N}\big)}_{\mathcal{L}_{\text{SIGReg}}}
$$

Untuk pelatihan aktual, dua hiperparameter trade-off diperkenalkan: $\lambda \in (0,1)$ mengatur keseimbangan antara tujuan pembentukan *embedding* kanonik dan pencegahan *collapse* (mengikuti peran $\lambda$ pada LeJEPA orisinal), sementara $\beta > 0$ mengatur bobot relatif $\mathcal{L}_{\text{semantic}}$ terhadap $\mathcal{L}_{\text{syntactic}}$ di dalam tujuan pembentukan *embedding* kanonik:

$$
\mathcal{L}_{\text{total}} = (1-\lambda)\big[\mathcal{L}_{\text{syntactic}} + \beta\, \mathcal{L}_{\text{semantic}}\big] + \lambda\, \mathcal{L}_{\text{SIGReg}}
$$

Komponen $\mathcal{L}_{\text{canonlen}}$ dioptimalkan secara terpisah (dengan bobot tersendiri atau dijumlahkan langsung, karena skalanya independen dari ruang *embedding* $d$-dimensi) dan tidak memengaruhi gradien $\theta, \phi$ secara langsung — hanya memperbarui $\psi$ — sehingga tidak disertakan dalam trade-off $\lambda,\beta$ di atas.

---

## 5. Sintesis dan Argumen Desain

Bagian ini merangkum implikasi dari rumusan di atas sebagai proposisi desain yang saling berkaitan.

**(a) Peran anchor vs. rata-rata.** *Anchor* $z_{n,1}$ dihitung dari keluaran *encoder* atas teks kanonik, dan seluruh view lain didekatkan ke anchor tersebut melalui *decoder* yang berperan sebagai *inverse mapping* (dari ruang variasi permukaan ke ruang kanonik). Ini berbeda secara mendasar dari LeJEPA citra, yang menggunakan rata-rata antar-view sebagai target karena tidak adanya *view* yang secara definisi lebih "asli" dari yang lain.

**(b) Pemisahan ruang regularisasi dan ruang penerjemahan.** SIGReg diterapkan pada keluaran *encoder* mentah untuk mencegah *collapse*, sedangkan *decoder* murni berperan sebagai penerjemah ke ruang kanonik. Jika *decoder* berhasil menerjemahkan bentuk non-kanonik secara akurat, ia akan "mendarat" pada wilayah ruang laten yang sama dengan keluaran *encoder* atas teks kanonik — wilayah yang sudah tunduk pada batasan distribusi Gaussian isotropik lewat SIGReg — sehingga sifat non-degeneratif terwarisi tanpa regularisasi ganda.

**(c) Non-interferensi antara loss semantik dan SIGReg.** Karena keduanya menegakkan properti statistik yang berbeda (bentuk distribusi marginal vs. struktur relasi berpasangan) pada ruang representasi yang berbeda pula (encoder vs. *pooled decoder output*), keduanya secara aljabar tidak saling meniadakan gradiennya.

**(d) Determinisme dan penanganan ketidakpastian aleatorik secara implisit.** Karena tidak tersedia sinyal supervisi eksplisit untuk menandai data yang bersifat aleatorik (ambigu), T-LeJEPA dirancang deterministik. Ambiguitas semacam itu ditangani secara implisit melalui kapasitas *encoder* bidireksional dalam mengintegrasikan konteks N-gram di sekitar setiap token, alih-alih lewat pemodelan distribusi eksplisit atas kemungkinan makna.

**(e) Pembagian peran sintaktik–semantik.** $\mathcal{L}_{\text{syntactic}}$ menjamin invariansi struktural — seluruh variasi permukaan suatu kalimat konvergen ke satu titik. $\mathcal{L}_{\text{semantic}}$ menjamin bahwa geometri relatif ruang laten antar-kalimat *yang berbeda* konsisten dengan derajat kemiripan maknanya, sebagaimana didefinisikan model guru eksternal. Kedua *loss* ini menjawab dua sumbu pemahaman bahasa yang dibahas pada Bagian 2.2 secara terpisah namun saling melengkapi.

---

## 6. Ringkasan Notasi

| Notasi | Deskripsi |
|---|---|
| $N$ | Jumlah titik data (kalimat unik) dalam korpus/batch |
| $V$ | Jumlah view per titik data (1 kanonik + $V-1$ non-kanonik) |
| $x_1^{(i)}$ | Bentuk kanonik kalimat ke-$i$ |
| $x_v^{(i)}, v>1$ | Bentuk non-kanonik ke-$v$ dari kalimat ke-$i$ |
| $L^{(i,j)}$ | Panjang token bentuk ke-$j$ dari data ke-$i$ |
| $f_\theta$ | Text encoder (Transformer encoder, bidireksional, *weight-shared* antar-view) |
| $g_\phi$ | Canonical decoder (Transformer decoder) |
| $\tau_\psi$ | Length predictor (feed-forward) |
| $z_{n,v}$ | Keluaran encoder untuk view $v$ data $n$ |
| $z_{c,n,v}$ | Keluaran decoder (representasi kanonik hasil terjemahan) untuk view $v$ data $n$ |
| $\mu_L(\cdot)$ | Operator mean-pooling sepanjang sumbu sekuens |
| $s_{\text{teacher}}$ | Skor kemiripan semantik dari model guru eksternal (mis. SimCSE/SBERT) |
| $A$ | Himpunan arah proyeksi acak untuk SIGReg |
| $T(\cdot)$ | Statistik uji normalitas berbasis fungsi karakteristik (mis. Epps–Pulley) |
| $\lambda, \beta$ | Hiperparameter trade-off antar-komponen loss |

---

## 7. Keterbatasan dan Agenda Validasi Empiris

Sebagai kejujuran ilmiah yang perlu dinyatakan eksplisit pada tahap metodologi:

1. **Ketergantungan pada model guru semantik.** Kualitas $\mathcal{L}_{\text{semantic}}$ bergantung penuh pada kualitas $s_{\text{teacher}}$; bias atau keterbatasan model guru (mis. SimCSE/SBERT) akan terwariskan ke T-LeJEPA. Studi ablasi terhadap pilihan model guru menjadi agenda validasi yang diperlukan.
2. **Bias minibatch pada SIGReg.** Sebagaimana dicatat Balestriero & LeCun (2025) pada LeJEPA orisinal, estimasi SIGReg berbasis minibatch memiliki bias berorde $O(1/N)$; efek ini pada domain teks (dengan panjang sekuens variabel) belum diuji secara empiris.
3. **Interaksi $\lambda$ dan $\beta$.** Rumusan Bagian 4.6 memperkenalkan dua hiperparameter trade-off; sensitivitas performa hilir terhadap keduanya, serta kemungkinan menyatukannya menjadi satu hiperparameter (semangat utama LeJEPA orisinal yang menekankan "single trade-off hyperparameter"), memerlukan studi tersendiri.
4. **Validasi empiris menyeluruh** — kurva pelatihan, evaluasi *downstream* (retrieval, STS, robustness terhadap teks non-standar lintas dialek/bahasa), serta perbandingan terhadap baseline (SimCSE, SBERT, dan varian LeJEPA yang diadaptasi naif ke teks) — merupakan kelanjutan wajib dari kerja metodologis ini sebelum klaim keunggulan empiris dapat diajukan.

---

## Daftar Pustaka

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 15619–15629.

Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv preprint arXiv:2511.08544*.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *International Conference on Learning Representations (ICLR)*.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *Proceedings of the 37th International Conference on Machine Learning (ICML)*.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*.

Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 6894–6910.

Gu, J., Bradbury, J., Xiong, C., Li, V. O. K., & Socher, R. (2018). Non-Autoregressive Neural Machine Translation. *International Conference on Learning Representations (ICLR)*.

Han, B., & Baldwin, T. (2011). Lexical Normalisation of Short Text Messages: Makn Sens a #Twitter. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (ACL-HLT)*, 368–378.

Han, B., Cook, P., & Baldwin, T. (2013). Lexical Normalization for Social Media Text. *ACM Transactions on Intelligent Systems and Technology*, 4(1), Article 5.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Deep Learning and Representation Learning Workshop*.

Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5574–5584.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview preprint*.

Navigli, R. (2009). Word Sense Disambiguation: A Survey. *ACM Computing Surveys*, 41(2), Article 10.

Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational Knowledge Distillation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 3967–3976.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982–3992.

Tung, F., & Mori, G. (2019). Similarity-Preserving Knowledge Distillation. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1365–1374.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 12310–12320.
