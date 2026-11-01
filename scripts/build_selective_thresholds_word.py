from __future__ import annotations

import re
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.shared import OxmlElement as SharedOxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from lxml import etree
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper" / "selective_thresholds_imbalance_long.tex"
WORD_DIR = ROOT / "paper" / "output" / "word"
FIG_DIR = WORD_DIR / "generated_figures"
OUT_PATH = WORD_DIR / "Selective_Thresholds_Imbalance_Academic_Final.docx"
MATHML2OMML_PATH = Path("/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl")


def prepare_tex_for_word(tex: str) -> str:
    """Keep the expanded paper logic while moving appendix-like detail out of the deliverable."""
    literature = r"""\section{Literature Review}

\subsection{Selective classification and class imbalance}

Chow's reject rule formalized the trade-off between recognition error and rejection cost \cite{chow1970reject}. Contemporary selective classification evaluates a predictor together with a selection function and studies selective risk against coverage \cite{geifman2017selective,hendrickx2024reject,franc2023optimal}. Under class imbalance, this trade-off is more delicate because aggregate accuracy can improve while minority-class utility falls. Recent work on long-tail rejection and precision--recall reject curves motivates reporting class-conditional diagnostics rather than only an accepted-case risk \cite{narasimhan2024longtail,fischer2024reject,shang2023precision}.

\subsection{Post-hoc thresholds and calibration}

The closest methodological precedent is AUC-based selective classification (AUCROSS), which rejects a score interval to improve ranking-oriented selective performance \cite{pugnana2023auc}. Our study differs by optimizing balanced accuracy on accepted cases and by treating a validation-optimized single threshold as the main baseline. Calibration is related but not sufficient: monotone calibration changes numerical threshold values without changing the order of accepted cases, whereas isotonic calibration can introduce ties that alter feasible reject intervals \cite{guo2017calibration,fisch2022calibrated,cattelan2024confidence}. We therefore study score-interval geometry directly and evaluate whether its gains survive same-coverage reject baselines.

"""
    lit_start = tex.find(r"\section{Literature Review}")
    method_start = tex.find(r"\section{Method}")
    if lit_start != -1 and method_start != -1:
        tex = tex[:lit_start] + literature + tex[method_start:]

    removals = [
        (r"\subsection{Best observed configurations}", r"\subsection{Default versus optimized single thresholds}"),
        (r"\subsection{Threshold-grid sensitivity}", r"\section{Discussion}"),
        (r"\subsection{Symmetric versus asymmetric rejection}", r"\subsection{Matched coverage}"),
    ]
    for start, end in removals:
        start_idx = tex.find(start)
        end_idx = tex.find(end)
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            tex = tex[:start_idx] + tex[end_idx:]
    tex = tex.replace(
        r"\subsection{Class-conditional coverage and calibration audit}",
        r"\subsection{Class-conditional coverage}",
    )
    cal_start = tex.find("The full-BAF calibration audit")
    cal_end = tex.find(r"\FloatBarrier", cal_start)
    if cal_start != -1 and cal_end != -1:
        tex = tex[:cal_start] + tex[cal_end + len(r"\FloatBarrier") :]
    class_table_start = tex.find(r"\begin{table}[htbp]", tex.find(r"\subsection{Class-conditional coverage"))
    class_table_end = tex.find(r"\end{table}", class_table_start)
    if class_table_start != -1 and class_table_end != -1:
        tex = tex[:class_table_start] + tex[class_table_end + len(r"\end{table}") :]
    ratio_table_start = tex.find(r"\begin{table}[htbp]", tex.find(r"\subsection{Effect of controlled training imbalance}"))
    ratio_table_end = tex.find(r"\end{table}", ratio_table_start)
    if ratio_table_start != -1 and ratio_table_end != -1:
        tex = tex[:ratio_table_start] + tex[ratio_table_end + len(r"\end{table}") :]
    replacements = {
        "A sensitivity analysis repeats representative BAF and Online Shoppers conditions with 41 and 101 quantiles.": "",
        "Ties prefer greater coverage and then the narrower score interval. Since $b$ is a maximum, not a forced rejection rate, this policy can use less rejection than its symmetric counterpart.": "Ties prefer greater coverage and then the narrower score interval.",
        "The symmetric lower and upper thresholds are inserted into the candidate set, guaranteeing at least one feasible interval with exactly the same validation coverage. The matched search then maximizes accepted-case balanced accuracy, breaking ties by the smallest coverage gap and then interval width. This isolates reject allocation more directly than the primary maximum-budget comparison.": "The symmetric thresholds are included in the candidate set, guaranteeing feasibility. The matched search maximizes accepted-case balanced accuracy and isolates allocation more directly than the maximum-budget comparison.",
        "To describe reject allocation, let $r_L$ and $r_U$ be validation rejection fractions below and above $t^{*}$. We define allocation asymmetry as $|r_L-r_U|/(r_L+r_U)$ when rejection is nonzero. A value of zero is evenly allocated; one places all rejected mass on one side.\n\n": "",
        "Logistic regression uses the L-BFGS solver and at most 1,000 iterations. LightGBM uses 160 trees, learning rate 0.05, 31 leaves, minimum child size 20, 0.9 row subsampling, and 0.9 feature subsampling. The MLP has hidden layers of 64 and 32 rectified-linear units, Adam optimization, batch size 256, $L_2$ coefficient $10^{-4}$, an initial learning rate of $10^{-3}$, and early stopping within 60 epochs.\n\n": "",
        "A fixed 50,000-row stratified BAF sample created with seed 7 supports controlled training-ratio experiments while retaining the same temporal partition logic.\n\n": "A fixed 50,000-row stratified BAF sample supports controlled training-ratio experiments while retaining the same temporal partition logic.\n\n",
        "Given only 21 positive test cases per split, SECOM is treated as a stress test rather than the headline source of evidence.": "Given only 21 positive test cases per split, SECOM is treated as a small-sample stress test.",
        "The controlled ratios are directly comparable with one another because their samples sizes are fixed.": "The controlled ratios are directly comparable with one another because their sample sizes are fixed.",
        "Units within a dataset share observations and are not independent dataset replications. The intervals therefore describe variability across the executed configuration matrix and are not population-level significance tests or guarantees for future datasets.": "Units within a dataset share observations, so the intervals describe stability across the executed configuration matrix rather than population-level guarantees.",
        "Curves aggregate all experimental units and should not be compared directly with the selected best-configuration values in Table~\\ref{tab:best}; vertical scales differ by dataset.": "Curves aggregate all experimental units; vertical scales differ by dataset.",
        "The complete 1\\%, 5\\%, and 10\\% tables are provided in the accompanying result artifacts; the 10\\% point is the most informative because the earlier budget analysis shows that this is where rejection has enough mass to matter.": "The 10\\% point is emphasized because the budget analysis shows that rejection has enough mass to matter there.",
        "Table~\\ref{tab:classcov} makes the cost of rejection explicit.": "The class-conditional coverage check makes the cost of rejection explicit.",
        "Table~\\ref{tab:ratios} answers the class-ratio research question using only controlled regimes with fixed training totals.": "The controlled-ratio analysis answers the class-ratio research question using only fixed-size training regimes.",
        "Paired-unit uncertainty intervals and threshold-grid sensitivity analyses quantify the stability and limits of the findings.": "Paired-unit uncertainty intervals and stability analyses quantify the limits of the findings.",
    }
    for old, new in replacements.items():
        tex = tex.replace(old, new)
    tex = tex.replace(
        "Paired-unit uncertainty intervals and threshold-grid sensitivity analyses quantify the stability and limits of the findings.",
        "Paired-unit uncertainty intervals and stability analyses quantify the limits of the findings.",
    )
    return compress_manuscript(tex)


def compress_manuscript(tex: str) -> str:
    """Remove repeated narration while preserving the original and added evidence."""

    def labelled_environment(label: str, environment: str) -> str:
        marker = rf"\label{{{label}}}"
        label_pos = tex.find(marker)
        if label_pos == -1:
            raise ValueError(f"Missing label: {label}")
        start = tex.rfind(rf"\begin{{{environment}}}", 0, label_pos)
        end_marker = rf"\end{{{environment}}}"
        end = tex.find(end_marker, label_pos)
        if start == -1 or end == -1:
            raise ValueError(f"Could not extract {environment}: {label}")
        return tex[start : end + len(end_marker)]

    def replace_section(start_heading: str, end_heading: str, content: str) -> None:
        nonlocal tex
        start = tex.find(start_heading)
        end = tex.find(end_heading, start)
        if start == -1 or end == -1:
            raise ValueError(f"Could not replace section between {start_heading} and {end_heading}")
        tex = tex[:start] + content.strip() + "\n\n" + tex[end:]

    data_table = labelled_environment("tab:data", "table")
    single_table = labelled_environment("tab:single", "table")
    paired_table = labelled_environment("tab:paired", "table")
    matched_table = labelled_environment("tab:matched", "table")
    baseline_table = labelled_environment("tab:baselines", "table")
    budget_figure = labelled_environment("fig:budgets", "figure")

    abstract = r"""\noindent\textbf{Abstract.}
Selective classifiers can abstain on uncertain cases, but under class imbalance a useful reject rule must be compared with a strong full-coverage threshold and with other rules at the same coverage. This study evaluates symmetric and asymmetric two-threshold policies using logistic regression, LightGBM, and a multilayer perceptron on Bank Account Fraud, Online Shoppers, and SECOM at 1\%, 5\%, and 10\% budgets. The primary outcome is balanced selective risk, supplemented by macro-F1, effective minority recall, and class-conditional coverage. At 10\%, dual thresholds reduce risk by 0.018--0.027 on the Bank Account Fraud settings and about 0.029 on Online Shoppers; SECOM is inconclusive. Random, confidence, margin, AUCROSS-style, and learned-error rejectors are then compared at matched coverage. AUCROSS is competitive, but none consistently improves on symmetric boundary rejection. Boundary error enrichment is higher on the larger tasks that benefit, although it is not a reliable unit-level predictor. The evidence therefore supports boundary-centered abstention when validation errors concentrate near an optimized decision threshold, while offering no stable advantage for asymmetric allocation or for imbalance severity alone.

\vspace{5pt}
\noindent\textbf{Keywords:} selective classification; reject option; class imbalance; dual thresholds; abstention
"""
    abstract_start = tex.find(r"\noindent\textbf{Abstract.}")
    intro_start = tex.find(r"\section{Introduction}")
    tex = tex[:abstract_start] + abstract + "\n" + tex[intro_start:]

    introduction = r"""\section{Introduction}

Imbalanced classifiers are often judged by a threshold that is poorly aligned with minority-class utility. A fixed 0.5 cut point may hide weak sensitivity, while resampling or class weighting can alter predicted-risk calibration without improving decisions \cite{vandengoorbergh2022harm,carriero2025harms}. This matters in fraud screening, purchasing-intention prediction, and manufacturing inspection, where rare positives can carry much of the operational value. Selective classification offers a complementary option: retain the fitted score, predict only on accepted cases, and defer a bounded subset for later handling.

For a scalar positive-class score, an optimized single threshold labels every case, whereas two thresholds create negative, reject, and positive regions. The practical question is not whether reject intervals exist, but whether they improve balanced selective risk beyond an optimized single threshold and beyond other rejectors at the same coverage. Comparing a dual rule only with the default 0.5 threshold would confound threshold correction with selective prediction; comparing policies at different realized coverage would confound reject placement with reject quantity. We therefore ask whether unequal allocation adds value after coverage matching and whether validation errors near the optimized boundary indicate when abstention will help.

We make three contributions. First, we compare validation-optimized single, symmetric dual, and asymmetric dual policies at 1\%, 5\%, and 10\% budgets across imbalanced tabular settings, including controlled class-ratio regimes. Second, matched-coverage experiments separate the value of a reject interval from asymmetric flexibility and compare random, confidence/margin, AUCROSS-style, and learned-error rejection using the same model scores and validation operating points. Third, boundary and reject error enrichment connect test gains to observable validation geometry. All comparisons report balanced selective risk with macro-F1, effective minority recall, and class-conditional coverage so rejected positives do not disappear from evaluation. The intended contribution is an empirical decision rule and a clear limit on asymmetric thresholds, not a claim that interval rejection itself is new.
"""
    literature = r"""\section{Literature Review}

Chow's reject rule formalized the error--rejection trade-off \cite{chow1970reject}; modern selective classification studies a predictor jointly with a selection function through risk and coverage \cite{geifman2017selective,hendrickx2024reject,franc2023optimal}. Learned rejectors, calibrated selectors, and guarantee-oriented methods address different deployment assumptions, but they share the idea that lower coverage should concentrate errors in the rejected subset. Our setting is deliberately post-hoc: a scalar score and fitted classifier already exist, and only the decision policy changes.

Class imbalance complicates selective evaluation because accepted-case accuracy or risk can improve while minority cases are rejected disproportionately. Long-tail rejection derives imbalance-aware selectors, while precision--recall reject curves expose class-specific changes as coverage falls \cite{narasimhan2024longtail,fischer2024reject,shang2023precision}. These results motivate balanced risk, class-conditional coverage, and effective minority recall in the same analysis. They also motivate retaining natural test prevalence when the training class ratio is experimentally controlled.

AUCROSS is the closest interval-rejection precedent, but it optimizes ranking performance rather than accepted-case balanced accuracy \cite{pugnana2023auc}. Our design instead treats a validation-optimized single threshold as the principal baseline and compares reject sets at matched coverage. Fixed 1\%, 5\%, and 10\% budgets answer an operational question rather than claiming a complete ranking over every coverage; the accompanying curve is therefore interpreted as descriptive rather than a universal summary.

Calibration remains a sensitivity condition rather than a solution. Monotone transforms preserve score order and hence the identity of rank-defined reject sets, whereas isotonic calibration can create large tie blocks and change which intervals satisfy a strict budget \cite{guo2017calibration,fisch2022calibrated,cattelan2024confidence}. This distinction is especially relevant for threshold policies: better probability calibration does not necessarily imply better localization of classification errors.
"""
    replace_section(r"\section{Introduction}", r"\section{Literature Review}", introduction)
    replace_section(r"\section{Literature Review}", r"\section{Method}", literature)

    method = r"""\section{Method}

\subsection{Decision policies}

Let $s(x)\in[0,1]$ be the positive-class score. The default policy uses 0.5; the optimized single policy selects
\begin{equation}
 t^{*}=\arg\max_{t\in\mathcal{T}} \operatorname{BA}\!\left(y,\mathbb{1}[s(x)\geq t]\right),
 \label{eq:single}
\end{equation}
where $\mathcal{T}$ contains 201 empirical score quantiles and outer endpoints. Threshold selection uses validation data only, and the untouched test set is evaluated once after the policy is fixed. This optimized full-coverage rule is the baseline for every reported risk difference.

The dual policy returns
\begin{equation}
 d(x)=
 \begin{cases}
 0, & s(x)<t_L,\\
 \bot, & t_L\leq s(x)<t_H,\\
 1, & s(x)\geq t_H,
 \end{cases}
 \label{eq:dual}
\end{equation}
for budgets $b\in\{0.01,0.05,0.10\}$. Symmetric rejection allocates the permitted validation mass approximately equally below and above $t^{*}$ using score order statistics. Symmetry refers to probability mass, not equal numerical distance from the threshold. Complete score-tie blocks are indivisible, so realized rejection never exceeds the budget and can be lower after isotonic calibration. Asymmetric search selects any lower and upper score quantiles within the budget that maximize accepted-case balanced accuracy, with ties favoring greater coverage and then a narrower interval.

To isolate allocation, matched asymmetric search targets the symmetric policy's realized rejection rate $r_S$:
\begin{equation}
 |r_A-r_S|\leq0.002.
 \label{eq:matched}
\end{equation}
The symmetric interval is included among candidates, guaranteeing feasibility. Matching occurs on validation data; small test-coverage differences can remain under sampling or temporal shift and are reported explicitly. At the same validation coverage, we also test random rejection, rejection nearest 0.5 (maximum-probability/margin), an AUCROSS-style interval maximizing accepted-case AUC \cite{pugnana2023auc}, and a quadratic logistic error model that rejects cases with the highest predicted optimized-single error. These baselines reuse the same fitted score, split, seed, and accepted-case threshold. No reject set is tuned on test outcomes.

\subsection{Evaluation metrics}

For accepted set $A$, coverage is $C=|A|/n$ and the primary outcome is
\begin{equation}
 \bsr(A)=1-\frac{1}{2}\left(
 \frac{\mathrm{TP}_A}{\mathrm{TP}_A+\mathrm{FN}_A}+
 \frac{\mathrm{TN}_A}{\mathrm{TN}_A+\mathrm{FP}_A}
 \right).
 \label{eq:bsr}
\end{equation}
Lower values are better. A missing accepted class receives zero recall, preventing a class-empty accepted subset from appearing optimal. For paired comparisons, $\Delta$BSR is policy BSR minus optimized-single BSR; negative values favor the selective policy. Because BSR excludes rejected cases, we additionally report macro-F1, class-conditional coverage $C_{+}$ and $C_{-}$, and effective minority recall
\begin{equation}
 \emr=\frac{\mathrm{TP}_A}{N_{+}},
 \label{eq:emr}
\end{equation}
where $N_{+}$ includes rejected positives.

For optimized-single error indicator $E$ and reject-set indicator $R$, reject error enrichment is
\begin{equation}
 \operatorname{REE}(R)=\frac{P(E=1\mid R=1)}{P(E=1)}.
 \label{eq:ree}
\end{equation}
Boundary error enrichment (BEE) applies the same ratio to the symmetric boundary band. Values above one indicate that the rejected region contains more errors than random selection would. Unlike a learned error rejector, BEE requires no additional flexible error model; it tests the specific geometric premise of dual thresholds before asymmetric allocation is considered. We examine both its dataset-level alignment with test gains and its limitations as a configuration-level predictor.

\subsection{Models and preprocessing}

Logistic regression, LightGBM \cite{ke2017lightgbm}, and a two-layer MLP cover linear, tree, and neural scores. LightGBM uses 160 trees and learning rate 0.05; the MLP uses 64- and 32-unit hidden layers with early stopping. Missing values are imputed and categorical features one-hot encoded using training data only; numeric features are standardized for logistic regression and the MLP. Full BAF also evaluates uncalibrated, Platt, isotonic, and temperature-scaled scores fitted on a dedicated calibration month. Other experiments use uncalibrated scores so the controlled-ratio matrix remains computationally bounded.
"""
    replace_section(r"\section{Method}", r"\section{Experimental Design}", method)

    experimental = rf"""\section{{Experimental Design}}

\subsection{{Datasets and partitions}}

Table~\ref{{tab:data}} covers three problem families spanning 1,567 to one million observations and positive prevalence from 1.103\% to 15.474\%. Full BAF uses months 0--4 for training, month 5 for calibration, month 6 for threshold selection, and month 7 for testing, preserving temporal order. The fixed BAF-50k sample supports controlled-ratio analysis with the same logic. Online Shoppers uses stratified 60/15/15/10\% train/calibration/selection/test splits. SECOM uses three stratified holdouts and has only 21 positive test cases per split, so it is a small-sample stress test rather than headline evidence.

{data_table}

\subsection{{Controlled factors and uncertainty}}

Seeds 7, 17, and 177 are used throughout. Fixed-size training ratios are 1:3--1:50 for BAF-50k and 1:3--1:10 for Online Shoppers and SECOM; only model-training data change, while validation and test prevalence remain natural. Comparisons within a controlled series isolate class ratio at fixed sample size, whereas comparison with the larger natural regime is descriptive. Full BAF crosses three models, three seeds, and four calibration conditions. Paired units share dataset, model, seed, training regime, and calibration. Percentile 95\% intervals use 5,000 paired-unit bootstrap resamples. Because configurations within a dataset share observations, these intervals describe stability in the executed matrix, not independent-dataset significance.
"""
    replace_section(r"\section{Experimental Design}", r"\section{Results}", experimental)

    when_table = r"""\begin{table}[htbp]
\centering
\caption{Validation error enrichment and test risk at the 10\% operating point.}
\label{tab:when}
\footnotesize
\begin{tabular}{lrrrrr}
\toprule
Dataset & Sym. $\Delta$BSR & BEE & AUCROSS $\Delta$BSR & Error $\Delta$BSR & Error REE \\
\midrule
BAF full & $-0.024$ & 2.278 & $-0.015$ & $\phantom{-}0.120$ & 4.325 \\
BAF 50k & $-0.018$ & 1.939 & $-0.009$ & $\phantom{-}0.093$ & 3.693 \\
Online & $-0.030$ & 2.680 & $-0.027$ & $\phantom{-}0.032$ & 2.729 \\
SECOM & $-0.004$ & 1.583 & $\phantom{-}0.000$ & $\phantom{-}0.012$ & 3.164 \\
\bottomrule
\end{tabular}
\end{table}"""

    when_figure = r"""\begin{figure}[htbp]
\centering
\caption{Dataset-level boundary error enrichment and symmetric dual-threshold test gain at the 10\% operating point.}
\label{fig:bee}
\end{figure}"""

    class_coverage_table = r"""\begin{table}[htbp]
\centering
\caption{Mean class-conditional coverage and effective minority recall at 10\%.}
\label{tab:classcov}
\footnotesize
\begin{tabular}{llrrr}
\toprule
Dataset & Policy & $C_{+}$ & $C_{-}$ & EMR \\
\midrule
BAF full & Optimized single & 1.000 & 1.000 & 0.783 \\
         & Symmetric dual & 0.910 & 0.920 & 0.730 \\
         & Asymmetric dual & 0.895 & 0.913 & 0.714 \\
BAF 50k & Optimized single & 1.000 & 1.000 & 0.702 \\
        & Symmetric dual & 0.913 & 0.910 & 0.653 \\
        & Asymmetric dual & 0.915 & 0.919 & 0.647 \\
Online & Optimized single & 1.000 & 1.000 & 0.797 \\
       & Symmetric dual & 0.886 & 0.909 & 0.733 \\
       & Asymmetric dual & 0.879 & 0.910 & 0.719 \\
SECOM & Optimized single & 1.000 & 1.000 & 0.438 \\
      & Symmetric dual & 0.901 & 0.913 & 0.393 \\
      & Asymmetric dual & 0.954 & 0.956 & 0.406 \\
\bottomrule
\end{tabular}
\end{table}"""

    results = rf"""\section{{Results}}

\subsection{{Strong single-threshold baseline}}

Optimizing the full-coverage threshold lowers mean BSR on every dataset (Table~\ref{{tab:single}}), by 0.040 on SECOM and 0.093--0.296 elsewhere. The selected cut points are far below 0.5 because balanced accuracy gives both classes equal weight. This correction generally raises minority recall, although macro-F1 can move differently because it is not the optimization target. Abstention is therefore evaluated against $t^*$, not against the default rule.

{single_table}

\subsection{{Dual thresholds and abstention budget}}

At 10\%, symmetric and asymmetric dual thresholds reduce BSR by 0.018--0.030 on full BAF, BAF-50k, and Online Shoppers, with paired intervals below zero (Table~\ref{{tab:paired}}). Similar results on full BAF and its fixed 50k sample indicate that the effect is not solely due to million-row scale. SECOM changes are small and both intervals cross zero, consistent with unstable threshold estimation from few failures. Figure~\ref{{fig:budgets}} shows that 1\% gains are minor and the clearer improvements arise at 5--10\%. On full BAF, for example, symmetric risk falls from 0.195 to 0.172 at 10\%, while EMR falls from 0.783 to 0.730; selective risk must therefore be read with coverage.

{paired_table}

{budget_figure}

\subsection{{Matched coverage and reject allocation}}

The maximum-budget comparison can mix allocation with reject quantity because either tie-safe policy may underuse its budget. Table~\ref{{tab:matched}} removes this difference. Validation gaps are at most 0.0004, test gaps at most 0.0019, and every risk-difference interval includes zero. The additional freedom to skew a fixed reject mass therefore has no stable benefit, even though asymmetric search contains the symmetric candidate and can look better on selection data.

{matched_table}

\subsection{{Same-coverage reject baselines}}

Table~\ref{{tab:baselines}} tests whether any 10\% rejection would produce the gain. Random rejection stays near the optimized-single result and is 0.014--0.027 worse than symmetric rejection on the larger settings. Maximum-probability/margin rejection is worse because it rejects near 0.5 rather than near $t^*$, which is often much lower under imbalance. The learned error model has high validation enrichment but often rejects positives and generalizes poorly, increasing BSR by 0.062--0.144 relative to the symmetric rule on the larger settings. AUCROSS-style rejection is the strongest competitor: its gap is only 0.003 on Online Shoppers but 0.008 on both BAF settings, and it does not consistently beat symmetric boundary rejection.

{baseline_table}

\subsection{{When diagnostics}}

BEE is 1.94--2.68 on the larger settings with stable gains, versus 1.58 on inconclusive SECOM (Table~\ref{{tab:when}}). This supports a screening rule: boundary rejection is more plausible when validation errors are enriched near $t^*$. The evidence is diagnostic rather than predictive. BEE does not reliably rank gains across individual configurations, and high REE alone is insufficient: the learned error model reaches REE values of 2.73--4.33 while worsening test BSR. Error concentration must therefore be paired with a stable, prespecified reject geometry and class-conditional checks.

{when_table}

{when_figure}

\subsection{{Imbalance and minority coverage}}

Under fixed-size controlled training ratios, dual thresholds improve BSR across BAF-50k and Online Shoppers ratios, but neither gain nor asymmetric advantage changes monotonically with imbalance. Symmetric $\Delta$BSR ranges from $-0.010$ to $-0.024$ on BAF-50k; asymmetric-minus-symmetric differences repeatedly change sign across both datasets. SECOM remains unstable, including a small asymmetric loss in one regime. At 10\%, positive coverage is generally no higher than negative coverage, and EMR falls on every dataset. Full-BAF asymmetric rejection, for example, accepts 89.5\% of positives versus 91.3\% of negatives and lowers EMR from 0.783 to 0.714. Thus class ratio alone does not answer ``when,'' and lower accepted-case risk does not mean that more positives are handled automatically.

{class_coverage_table}
"""
    replace_section(r"\section{Results}", r"\section{Discussion}", results)

    discussion = r"""\section{Discussion}

The experiments support a narrower and more useful claim than ``two thresholds are better than one.'' First optimize the ordinary threshold: otherwise threshold correction is confounded with abstention. A boundary-centered reject interval then improves balanced selective risk on the larger BAF and Online Shoppers settings at 5--10\% coverage loss, but not decisively on small SECOM. The 1\% point is often too small for stable improvement, especially when finite score grids or tie blocks prevent full use of the nominal budget.

Same-coverage comparisons identify the mechanism. Random and conventional confidence/margin rejection do not reproduce the gain, while AUCROSS is competitive but not consistently superior. Matched coverage removes the apparent advantage of asymmetric allocation. The benefit therefore comes mainly from placing a bounded reject region around the optimized decision boundary, not from abstention alone or from skewing rejection toward one side.

BEE offers a partial answer to ``when'': the successful larger tasks contain roughly twice or more the background error rate near $t^*$. It should be used as a diagnostic, not a guarantee, because unit-level association is imperfect and a flexible error rejector can overfit. A reasonable workflow is to estimate BEE on threshold-selection data, reject only when enrichment is material and stable across seeds, and verify test-independent class-conditional coverage. Deployment decisions must examine $C_{+}$, $C_{-}$, and EMR; accepted-case risk can fall while fewer positives are processed automatically.

Calibration and score ties clarify another boundary. Platt and temperature scaling preserve score order and hence rank-defined reject sets; isotonic calibration can collapse many scores into tied blocks and reduce feasible coverage resolution. Numerical calibration quality should therefore not be mistaken for selective ordering quality.

The evidence is limited to three binary tabular families, three seeds, finite threshold grids, and configuration units that share observations. Controlled ratios use BAF-50k rather than full BAF, and SECOM has few positive test cases. The bootstrap quantifies stability across the executed configurations rather than population-level significance. The study also leaves abstentions unresolved, so it does not model human-review cost, review accuracy, or end-to-end utility. Larger independent industrial data, nested resampling, distribution shift, and explicit deferral costs remain important extensions \cite{liang2024distribution,hallberg2025conformal}.
"""
    replace_section(r"\section{Discussion}", r"\section{Conclusion}", discussion)

    conclusion = r"""\section{Conclusion}

Two thresholds are most useful here when an optimized single threshold still has errors concentrated near its validation boundary and 5--10\% abstention is operationally acceptable. They reduce balanced selective risk on the larger BAF and Online Shoppers settings, but SECOM is inconclusive. Same-coverage baselines show that reject placement matters; matched coverage shows no stable asymmetric advantage; and controlled ratios show that imbalance severity alone is not predictive. The practical sequence is to optimize $t^*$, inspect BEE and class-conditional coverage, then validate a boundary interval at realistic budgets while tracking EMR.

\section*{Data and Code Availability}

BAF is documented by Jesus et al. \cite{jesus2022turning}; Online Shoppers and SECOM are available from UCI \cite{sakar2018online,mccann2008secom}. The accompanying code records splits, seeds, policy outputs, paired comparisons, and tie-safe threshold searches.

\section*{Appendix A: Supplementary Sensitivity Results}

Table~\ref{tab:ratios_appendix} gives the fixed-size training-ratio results summarized in Section 5.6. Table~\ref{tab:grid_appendix} checks whether the finite threshold grid changes the qualitative conclusion. These tables preserve the original experiment detail without interrupting the same-coverage and ``when'' analyses in the main text.

\begin{table}[htbp]
\centering
\caption{Mean 10\% risk differences under fixed-size controlled training ratios.}
\label{tab:ratios_appendix}
\footnotesize
\begin{tabular}{llrrr}
\toprule
Dataset & Ratio & Sym.-single & Asym.-single & Asym.-sym. \\
\midrule
BAF 50k & 1:3 & $-0.0210$ & $-0.0247$ & $-0.0038$ \\
        & 1:5 & $-0.0186$ & $-0.0165$ & $\phantom{-}0.0021$ \\
        & 1:10 & $-0.0242$ & $-0.0173$ & $\phantom{-}0.0069$ \\
        & 1:20 & $-0.0102$ & $-0.0103$ & $-0.0001$ \\
        & 1:50 & $-0.0122$ & $-0.0192$ & $-0.0070$ \\
Online & 1:3 & $-0.0287$ & $-0.0264$ & $\phantom{-}0.0023$ \\
       & 1:5 & $-0.0278$ & $-0.0342$ & $-0.0064$ \\
       & 1:10 & $-0.0320$ & $-0.0292$ & $\phantom{-}0.0028$ \\
SECOM & 1:3 & $-0.0018$ & $-0.0042$ & $-0.0025$ \\
      & 1:5 & $-0.0002$ & $-0.0017$ & $-0.0015$ \\
      & 1:10 & $-0.0159$ & $\phantom{-}0.0040$ & $\phantom{-}0.0199$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Threshold-grid sensitivity for budget-constrained asymmetric rejection.}
\label{tab:grid_appendix}
\footnotesize
\begin{tabular}{lrrrrr}
\toprule
Dataset & Grid & BSR (1\%) & Cov. (1\%) & BSR (5\%) & Cov. (5\%) \\
\midrule
BAF full & 41 & 0.1887 & 1.0000 & 0.1745 & 0.9549 \\
         & 101 & 0.1864 & 0.9910 & 0.1743 & 0.9546 \\
         & 201 & 0.1861 & 0.9909 & 0.1743 & 0.9546 \\
Online & 41 & 0.1578 & 1.0000 & 0.1461 & 0.9530 \\
       & 101 & 0.1566 & 0.9935 & 0.1379 & 0.9543 \\
       & 201 & 0.1549 & 0.9916 & 0.1459 & 0.9543 \\
\bottomrule
\end{tabular}
\end{table}
"""
    replace_section(r"\section{Conclusion}", r"\section*{Data and Code Availability}", conclusion)
    first_data = tex.find(r"\section*{Data and Code Availability}", tex.find(r"\section{Conclusion}") + 1)
    duplicate_data = tex.find(r"\section*{Data and Code Availability}", first_data + 1)
    bib_start = tex.find(r"\begin{thebibliography}", duplicate_data)
    if duplicate_data != -1 and bib_start != -1:
        # The compact conclusion already includes this section.
        tex = tex[:duplicate_data] + tex[bib_start:]
    return tex


def find_font(name: str = "Times New Roman.ttf") -> str | None:
    candidates = [
        Path("/System/Library/Fonts/Supplemental") / name,
        Path("/Library/Fonts") / name,
        Path("/System/Library/Fonts") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


FONT_REG = find_font("Times New Roman.ttf")
FONT_BOLD = find_font("Times New Roman Bold.ttf") or FONT_REG


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REG
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def clean_math(value: str) -> str:
    value = value.replace(r"\{", "§MLBRACE§").replace(r"\}", "§MRBRACE§")
    replacements = {
        r"\operatorname{BA}": "BA",
        r"\operatorname{BSR}": "BSR",
        r"\operatorname{REE}": "REE",
        r"\mathcal{T}": "𝒯",
        r"\arg\max": "arg max",
        r"\mathrm{TP}": "TP",
        r"\mathrm{FN}": "FN",
        r"\mathrm{TN}": "TN",
        r"\mathrm{FP}": "FP",
        r"\mathbb{1}": "𝟙",
        r"\bot": "⊥",
        r"\in": "∈",
        r"\mid": " | ",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\left": "",
        r"\right": "",
        r"\lfloor": "floor(",
        r"\rfloor": ")",
        r"\phantom{-}": "",
        r"\Delta": "Δ",
        r"\rho": "ρ",
        r"\rm": "",
        r"\%": "%",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = re.sub(r"\\frac\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\^\{\*\}", "*", value)
    value = re.sub(r"\^\{([^{}]+)\}", r"^\1", value)
    value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\\", "")
    value = value.replace("^*", "*")
    value = re.sub(r"_([AHLS])", lambda m: "_" + m.group(1).lower(), value)
    subscript_map = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
    value = re.sub(
        r"_([0123456789+\-=()aehijklmnoprstuvx]+)",
        lambda m: m.group(1).translate(subscript_map),
        value,
    )
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def strip_latex(value: str, cite_map: dict[str, int], ref_map: dict[str, str]) -> str:
    accent_replacements = {
        "L\\\"uth": "Lueth",
        "J\\\"ager": "Jaeger",
        "Szabadv\\'ary": "Szabadvary",
        "L\\\"ofstr\\\"om": "Lofstrom",
        "S\\\"onstr\\\"od": "Sonstrod",
    }
    for old, new in accent_replacements.items():
        value = value.replace(old, new)
    value = value.replace("~", " ")
    value = value.replace("``", '"').replace("''", '"')
    value = value.replace("---", "-").replace("--", "-")
    value = value.replace(r"\%", "%").replace(r"\$", "$")
    value = re.sub(r"\\ci\{\}", "95% CI", value)
    value = re.sub(r"\\ci\b", "95% CI", value)
    value = re.sub(r"\\bsr\b", "BSR", value)
    value = re.sub(r"\\emr\b", "EMR", value)
    value = re.sub(r"\\suppressfloats\[[^\]]+\]", "", value)
    value = re.sub(r"\\FloatBarrier", "", value)
    value = re.sub(r"\\label\{[^}]+\}", "", value)
    value = re.sub(r"\\url\{([^}]+)\}", r"\1", value)
    value = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(cite_map[k.strip()]) for k in m.group(1).split(",") if k.strip() in cite_map) + "]", value)
    value = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + ref_map.get(m.group(1), "?") + ")", value)
    value = re.sub(r"\\ref\{([^}]+)\}", lambda m: ref_map.get(m.group(1), "?"), value)
    value = re.sub(r"\$([^$]+)\$", lambda m: clean_math(m.group(1)), value)
    value = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\emph\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\item\s*", "", value)
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or "", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("§MLBRACE§", "{").replace("§MRBRACE§", "}")
    value = value.replace("DeltaBSR", "ΔBSR").replace("Delta BSR", "ΔBSR")
    value = value.replace("Delta C_", "ΔC_")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def collect_maps(tex: str) -> tuple[dict[str, int], dict[str, str]]:
    cite_map = {key: i + 1 for i, key in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", tex))}
    ref_map: dict[str, str] = {}
    table_no = 0
    figure_no = 0
    eq_no = 0
    token_re = re.compile(r"\\begin\{(table|figure|equation)\}(.*?)\\end\{\1\}", re.S)
    for match in token_re.finditer(tex):
        kind, block = match.group(1), match.group(2)
        label = re.search(r"\\label\{([^}]+)\}", block)
        if kind == "table":
            table_no += 1
            if label:
                ref_map[label.group(1)] = str(table_no)
        elif kind == "figure":
            figure_no += 1
            if label:
                ref_map[label.group(1)] = str(figure_no)
        else:
            eq_no += 1
            if label:
                ref_map[label.group(1)] = str(eq_no)
    ref_map["sec:coveragecal"] = "5.9"
    return cite_map, ref_map


def set_run_font(run, size: float, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_paragraph(doc: Document, text: str, size: float = 10.0, align=None, first_line: bool = True) -> None:
    if not text:
        return
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(1.4)
    pf.line_spacing = 1.0
    if first_line:
        pf.first_line_indent = Cm(0.45)
    run = p.add_run(text)
    set_run_font(run, size)


def add_caption(doc: Document, text: str, size: float = 9.5) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    if text.startswith("Table "):
        p.paragraph_format.keep_with_next = True
    m = re.match(r"^(Table|Figure) (\d+)\. (.*)$", text)
    if m:
        r1 = p.add_run(f"{m.group(1)} {m.group(2)}. ")
        set_run_font(r1, size, bold=True)
        r2 = p.add_run(m.group(3))
        set_run_font(r2, size)
    else:
        r = p.add_run(text)
        set_run_font(r, size)


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5 if level == 1 else 3)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    set_run_font(run, 11, bold=True, italic=(level > 1))


def remove_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "nil")


def set_cell_border(cell, **kwargs) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge, attrs in kwargs.items():
        tag = "w:" + edge
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        for key, value in attrs.items():
            element.set(qn("w:" + key), str(value))


def repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_cell_text(cell, text: str, size: float, bold: bool = False) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size, bold)


def add_word_table(doc: Document, caption: str, rows: list[list[str]], note: str | None, table_no: int) -> None:
    add_caption(doc, f"Table {table_no}. {caption}", 9.2)
    if not rows:
        return
    cols = max(len(r) for r in rows)
    normalized = [r + [""] * (cols - len(r)) for r in rows]
    table = doc.add_table(rows=len(normalized), cols=cols)
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.autofit = True
    remove_table_borders(table)
    size = 8.5 if cols >= 7 else 9.2
    for i, row in enumerate(normalized):
        for j, text in enumerate(row):
            set_cell_text(table.cell(i, j), text, size, bold=(i == 0))
        keep_row_together(table.rows[i])
    border = {"val": "single", "sz": "8", "space": "0", "color": "000000"}
    thin = {"val": "single", "sz": "4", "space": "0", "color": "000000"}
    for cell in table.rows[0].cells:
        set_cell_border(cell, top=border, bottom=thin)
    repeat_table_header(table.rows[0])
    for cell in table.rows[-1].cells:
        set_cell_border(cell, bottom=border)
    if note:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.right_indent = Cm(1.0)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(note)
        set_run_font(r, 7.6)


def parse_table(block: str, cite_map: dict[str, int], ref_map: dict[str, str]) -> tuple[str, list[list[str]], str | None]:
    caption_match = re.search(r"\\caption\{(.*?)\}", block, re.S)
    caption = strip_latex(caption_match.group(1), cite_map, ref_map) if caption_match else ""
    tabular_match = re.search(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", block, re.S)
    rows: list[list[str]] = []
    if tabular_match:
        raw = tabular_match.group(1)
        current = ""
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
                continue
            current += " " + line
            if "\\\\" in line:
                current = current.split("\\\\")[0]
                cells = [strip_latex(cell.strip(), cite_map, ref_map) for cell in current.split("&")]
                if cells and any(cells):
                    rows.append(cells)
                current = ""
    note_match = re.search(r"\\item\s+(.*?)(?:\\end\{tablenotes\}|\\end\{flushleft\})", block, re.S)
    if not note_match:
        note_match = re.search(r"\\begin\{flushleft\}\s*\\footnotesize\s+(.*?)\\end\{flushleft\}", block, re.S)
    note = strip_latex(note_match.group(1), cite_map, ref_map) if note_match else None
    return caption, rows, note


def mathml_to_omml(mathml: str):
    if not MATHML2OMML_PATH.exists():
        raise FileNotFoundError(f"Missing Word MathML transform: {MATHML2OMML_PATH}")
    transform = etree.XSLT(etree.parse(str(MATHML2OMML_PATH)))
    math_root = etree.fromstring(mathml.encode("utf-8"))
    return transform(math_root).getroot()


def add_equation(doc: Document, block: str, number: int) -> None:
    display_mathml = {
        1: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <msup><mi>t</mi><mo>*</mo></msup><mo>=</mo>
            <munder><mi mathvariant="normal">arg max</mi><mrow><mi>t</mi><mo>∈</mo><mi>𝒯</mi></mrow></munder>
            <mi mathvariant="normal">BA</mi><mo>(</mo><mi>y</mi><mo>,</mo><mi>𝟙</mi><mo>[</mo>
            <mi>s</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>≥</mo><mi>t</mi><mo>]</mo><mo>)</mo>
        </mrow></math>""",
        2: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi>d</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo>
            <mfenced open="{" close=""><mtable columnalign="left left">
                <mtr><mtd><mn>0</mn></mtd><mtd><mtext>if </mtext><mi>s</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>&lt;</mo><msub><mi>t</mi><mi>L</mi></msub></mtd></mtr>
                <mtr><mtd><mo>⊥</mo></mtd><mtd><mtext>if </mtext><msub><mi>t</mi><mi>L</mi></msub><mo>≤</mo><mi>s</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>&lt;</mo><msub><mi>t</mi><mi>H</mi></msub></mtd></mtr>
                <mtr><mtd><mn>1</mn></mtd><mtd><mtext>if </mtext><mi>s</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>≥</mo><msub><mi>t</mi><mi>H</mi></msub></mtd></mtr>
            </mtable></mfenced>
        </mrow></math>""",
        3: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mo>|</mo><msub><mi>r</mi><mi>A</mi></msub><mo>−</mo><msub><mi>r</mi><mi>S</mi></msub><mo>|</mo><mo>≤</mo><mn>0.002</mn>
        </mrow></math>""",
        4: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi mathvariant="normal">BSR</mi><mo>(</mo><mi>A</mi><mo>)</mo><mo>=</mo><mn>1</mn><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac>
            <mo>[</mo>
            <mfrac><msub><mi mathvariant="normal">TP</mi><mi>A</mi></msub><mrow><msub><mi mathvariant="normal">TP</mi><mi>A</mi></msub><mo>+</mo><msub><mi mathvariant="normal">FN</mi><mi>A</mi></msub></mrow></mfrac>
            <mo>+</mo>
            <mfrac><msub><mi mathvariant="normal">TN</mi><mi>A</mi></msub><mrow><msub><mi mathvariant="normal">TN</mi><mi>A</mi></msub><mo>+</mo><msub><mi mathvariant="normal">FP</mi><mi>A</mi></msub></mrow></mfrac>
            <mo>]</mo>
        </mrow></math>""",
        5: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi mathvariant="normal">EMR</mi><mo>=</mo>
            <mfrac><msub><mi mathvariant="normal">TP</mi><mi>A</mi></msub><msub><mi>N</mi><mo>+</mo></msub></mfrac>
        </mrow></math>""",
        6: """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>
            <mi mathvariant="normal">REE</mi><mo>(</mo><mi>R</mi><mo>)</mo><mo>=</mo>
            <mfrac>
                <mrow><mi>P</mi><mo>(</mo><mi>E</mi><mo>=</mo><mn>1</mn><mo>|</mo><mi>R</mi><mo>=</mo><mn>1</mn><mo>)</mo></mrow>
                <mrow><mi>P</mi><mo>(</mo><mi>E</mi><mo>=</mo><mn>1</mn><mo>)</mo></mrow>
            </mfrac>
        </mrow></math>""",
    }
    if number not in display_mathml:
        raise ValueError(f"No native Word equation defined for equation {number}")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    p.paragraph_format.tab_stops.add_tab_stop(Cm(7.8), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.6), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    p._p.append(mathml_to_omml(display_mathml[number]))
    number_run = p.add_run(f"\t({number})")
    set_run_font(number_run, 9.5)


def draw_dual_delta(path: Path) -> None:
    img = Image.new("RGB", (1500, 760), "white")
    d = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = 160, 90, 95, 125
    plot_w = img.width - margin_l - margin_r
    plot_h = img.height - margin_t - margin_b
    xmin, xmax = 1.4, 2.85
    ymin, ymax = -0.033, 0.003
    points = [
        (2.278, -0.0238, "BAF full", (16, -48)),
        (1.939, -0.0177, "BAF 50k", (-155, -6)),
        (2.680, -0.0295, "Online", (-120, -48)),
        (1.583, -0.0040, "SECOM", (18, -8)),
    ]

    def x(v: float) -> float:
        return margin_l + (v - xmin) / (xmax - xmin) * plot_w

    def y(v: float) -> float:
        return margin_t + (ymax - v) / (ymax - ymin) * plot_h

    d.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="black", width=2)
    d.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="black", width=2)
    d.line((margin_l, y(0), margin_l + plot_w, y(0)), fill=(100, 100, 100), width=2)
    for tick in [-0.03, -0.02, -0.01, 0.0]:
        yy = y(tick)
        d.line((margin_l - 8, yy, margin_l + plot_w, yy), fill=(225, 225, 225), width=1)
        d.text((55, yy - 14), f"{tick:.2f}", font=font(25), fill="black")
    for tick in [1.5, 2.0, 2.5]:
        xx = x(tick)
        d.line((xx, margin_t, xx, margin_t + plot_h + 8), fill=(225, 225, 225), width=1)
        d.text((xx - 20, margin_t + plot_h + 18), f"{tick:.1f}", font=font(25), fill="black")
    for bee, delta, label, offset in points:
        xx, yy = x(bee), y(delta)
        d.ellipse((xx - 14, yy - 14, xx + 14, yy + 14), fill=(32, 73, 111), outline="black", width=2)
        d.text((xx + offset[0], yy + offset[1]), label, font=font(25, True), fill="black")
    d.text((395, 20), "Boundary error enrichment and test gain", font=font(32, True), fill="black")
    d.text((470, img.height - 60), "Boundary error enrichment (validation)", font=font(25), fill="black")
    d.text((15, 315), "Δ BSR", font=font(25, True), fill="black")
    d.text((1020, 70), "Lower is better", font=font(23), fill=(80, 80, 80))
    img.save(path)


def draw_budget(path: Path) -> None:
    img = Image.new("RGB", (1500, 700), "white")
    d = ImageDraw.Draw(img)
    panels = [
        ("BAF full", (0.16, 0.20), [0.1954, 0.1928, 0.1841, 0.1715], [0.1954, 0.1929, 0.1833, 0.1683]),
        ("Online Shoppers", (0.15, 0.19), [0.1852, 0.1821, 0.1700, 0.1557], [0.1852, 0.1828, 0.1696, 0.1560]),
        ("SECOM", (0.432, 0.447), [0.4394, 0.4394, 0.4359, 0.4354], [0.4394, 0.4394, 0.4385, 0.4374]),
    ]
    budgets = [0, 1, 5, 10]
    colors = [(32, 73, 111), (190, 72, 45)]
    d.text((500, 18), "Mean BSR by abstention budget", font=font(32, True), fill="black")
    for pidx, (title, (ymin, ymax), sym, asym) in enumerate(panels):
        left = 85 + pidx * 470
        top = 95
        w, h = 385, 445
        d.rectangle((left, top, left + w, top + h), outline=(140, 140, 140), width=2)
        d.text((left + 115, top - 45), title, font=font(26, True), fill="black")
        def x(b: float) -> float:
            return left + 40 + b / 10 * (w - 75)
        def y(v: float) -> float:
            return top + 25 + (ymax - v) / (ymax - ymin) * (h - 60)
        for b in [0, 5, 10]:
            xx = x(b)
            d.line((xx, top + h - 35, xx, top + h - 25), fill="black", width=2)
            d.text((xx - 10, top + h - 18), str(b), font=font(20), fill="black")
        for series, color in [(sym, colors[0]), (asym, colors[1])]:
            pts = [(x(b), y(v)) for b, v in zip(budgets, series)]
            d.line(pts, fill=color, width=5)
            for xx, yy in pts:
                d.ellipse((xx - 7, yy - 7, xx + 7, yy + 7), fill=color)
        d.text((left + 130, top + h + 20), "Budget (%)", font=font(22), fill="black")
    d.rectangle((570, 595, 600, 618), fill=colors[0])
    d.text((612, 590), "Symmetric", font=font(24), fill="black")
    d.rectangle((760, 595, 790, 618), fill=colors[1])
    d.text((802, 590), "Asymmetric", font=font(24), fill="black")
    img.save(path)


def add_figure(doc: Document, caption: str, figure_no: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"figure_{figure_no}.png"
    is_budget = "budget" in caption.lower()
    if is_budget:
        draw_budget(path)
    else:
        draw_dual_delta(path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(5.9))
    if is_budget:
        caption = "Mean balanced selective risk by maximum abstention budget. The point at 0% is the optimized single threshold. Curves aggregate all experimental units; vertical scales differ by dataset."
    else:
        caption = "Dataset-level boundary error enrichment and symmetric dual-threshold test gain at 10%. Lower ΔBSR indicates greater benefit; four points are descriptive rather than a fitted predictive model."
    add_caption(doc, f"Figure {figure_no}. {caption}", 9.2)


def add_front_matter(doc: Document) -> None:
    title = "When Do Two Decision Thresholds Improve Selective Classification Under Class Imbalance?"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(title)
    set_run_font(r, 17, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Zizhen Wang")
    set_run_font(r, 11, bold=True)
    r2 = p.add_run("*")
    set_run_font(r2, 8)
    r2.font.superscript = True
    for line in [
        "University of Nottingham Ningbo China, Ningbo, Zhejiang 315100, China",
        "*Corresponding author: scyzw22@nottingham.edu.cn",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(line)
        set_run_font(r, 9.6)


def add_abstract_keywords(doc: Document, tex: str, cite_map: dict[str, int], ref_map: dict[str, str]) -> None:
    abstract = re.search(r"\\noindent\\textbf\{Abstract\.\}(.*?)(?:\\vspace\{5pt\})", tex, re.S).group(1)
    keywords = re.search(r"\\noindent\\textbf\{Keywords:\}(.*?)\\section\{Introduction\}", tex, re.S).group(1)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Abstract. ")
    set_run_font(r, 10.0, bold=True)
    r = p.add_run(strip_latex(abstract, cite_map, ref_map))
    set_run_font(r, 10.0)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Keywords: ")
    set_run_font(r, 10.0, bold=True)
    r = p.add_run(strip_latex(keywords, cite_map, ref_map))
    set_run_font(r, 10.0)


def add_body(doc: Document, tex: str, cite_map: dict[str, int], ref_map: dict[str, str]) -> None:
    body = tex[tex.index(r"\section{Introduction}") : tex.index(r"\begin{thebibliography}")]
    token_re = re.compile(r"(\\section\*?\{[^}]+\}|\\subsection\{[^}]+\}|\\begin\{equation\}.*?\\end\{equation\}|\\begin\{table\}.*?\\end\{table\}|\\begin\{figure\}.*?\\end\{figure\})", re.S)
    section_no = 0
    subsection_no = 0
    table_no = 0
    figure_no = 0
    equation_no = 0
    pos = 0
    no_indent_next = False
    for match in token_re.finditer(body):
        text_chunk = body[pos : match.start()]
        added = add_text_chunk(doc, text_chunk, cite_map, ref_map, first_para_no_indent=no_indent_next)
        if added:
            no_indent_next = False
        token = match.group(1)
        if token.startswith(r"\section"):
            title = re.search(r"\\section\*?\{([^}]+)\}", token).group(1)
            if token.startswith(r"\section*"):
                if title.startswith("Appendix"):
                    doc.add_page_break()
                add_heading(doc, title, 1)
            else:
                section_no += 1
                subsection_no = 0
                add_heading(doc, f"{section_no} {title}", 1)
            no_indent_next = True
        elif token.startswith(r"\subsection"):
            subsection_no += 1
            title = re.search(r"\\subsection\{([^}]+)\}", token).group(1)
            add_heading(doc, f"{section_no}.{subsection_no} {title}", 2)
            no_indent_next = True
        elif token.startswith(r"\begin{equation}"):
            equation_no += 1
            block = re.search(r"\\begin\{equation\}(.*?)\\end\{equation\}", token, re.S).group(1)
            add_equation(doc, block, equation_no)
        elif token.startswith(r"\begin{table}"):
            table_no += 1
            caption, rows, note = parse_table(token, cite_map, ref_map)
            add_word_table(doc, caption, rows, note, table_no)
        elif token.startswith(r"\begin{figure}"):
            figure_no += 1
            caption_match = re.search(r"\\caption\{(.*?)\}", token, re.S)
            caption = strip_latex(caption_match.group(1), cite_map, ref_map) if caption_match else ""
            add_figure(doc, caption, figure_no)
        pos = match.end()
    add_text_chunk(doc, body[pos:], cite_map, ref_map, first_para_no_indent=no_indent_next)


def add_text_chunk(
    doc: Document,
    chunk: str,
    cite_map: dict[str, int],
    ref_map: dict[str, str],
    first_para_no_indent: bool = False,
) -> int:
    chunk = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", chunk, flags=re.S)
    chunk = re.sub(r"\\begin\{threeparttable\}|\\end\{threeparttable\}", "", chunk)
    chunk = re.sub(r"\\begin\{tablenotes\}.*?\\end\{tablenotes\}", "", chunk, flags=re.S)
    paragraphs = re.split(r"\n\s*\n", chunk)
    added = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para = strip_latex(para, cite_map, ref_map)
        if para:
            add_paragraph(doc, para, first_line=not (first_para_no_indent and added == 0))
            added += 1
    return added


def add_references(doc: Document, tex: str, cite_map: dict[str, int], ref_map: dict[str, str]) -> None:
    add_heading(doc, "References", 1)
    bib = tex[tex.index(r"\begin{thebibliography}") :]
    items = re.split(r"\\bibitem\{([^}]+)\}", bib)[1:]
    for idx in range(0, len(items), 2):
        key = items[idx]
        content = items[idx + 1]
        content = content.split(r"\bibitem")[0]
        content = content.split(r"\end{thebibliography}")[0]
        text = strip_latex(content, cite_map, ref_map)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.55)
        p.paragraph_format.first_line_indent = Cm(-0.55)
        p.paragraph_format.space_after = Pt(1.2)
        r = p.add_run(f"[{cite_map[key]}] {text}")
        set_run_font(r, 7.7)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.start_type = WD_SECTION.NEW_PAGE
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(4.0)
    section.bottom_margin = Cm(2.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10.0)
    settings = doc.settings._element
    view = settings.find(qn("w:view"))
    if view is None:
        view = OxmlElement("w:view")
        settings.insert(0, view)
    view.set(qn("w:val"), "print")
    zoom = settings.find(qn("w:zoom"))
    if zoom is None:
        zoom = OxmlElement("w:zoom")
        settings.insert(1, zoom)
    zoom.set(qn("w:percent"), "100")


def add_footer_page_number(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    set_run_font(run, 9)

    fld_begin = SharedOxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = SharedOxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_separate = SharedOxmlElement("w:fldChar")
    fld_separate.set(qn("w:fldCharType"), "separate")
    fallback = SharedOxmlElement("w:t")
    fallback.text = "1"
    fld_end = SharedOxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_separate)
    run._r.append(fallback)
    run._r.append(fld_end)


def scrub_metadata(path: Path) -> None:
    # Keep Word metadata simple and avoid carrying TeX4ht conversion leftovers.
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"<dc:creator>.*?</dc:creator>", b"<dc:creator>Zizhen Wang</dc:creator>", data)
                data = re.sub(rb"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", b"<cp:lastModifiedBy>Zizhen Wang</cp:lastModifiedBy>", data)
            zout.writestr(item, data)
    tmp.replace(path)


def main() -> None:
    WORD_DIR.mkdir(parents=True, exist_ok=True)
    tex = prepare_tex_for_word(TEX_PATH.read_text(encoding="utf-8"))
    cite_map, ref_map = collect_maps(tex)
    doc = Document()
    configure_document(doc)
    add_front_matter(doc)
    add_abstract_keywords(doc, tex, cite_map, ref_map)
    add_body(doc, tex, cite_map, ref_map)
    add_references(doc, tex, cite_map, ref_map)
    doc.save(OUT_PATH)
    scrub_metadata(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
