\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{VITSCOPEThesis}[2026/02/13 VIT-Vellore SCOPE Thesis or Project report format]
\LoadClass[12pt]{report}

\RequirePackage[a4paper,left=1.5in,right=1in,top=1in,bottom=1in,nohead]{geometry}
\RequirePackage{setspace}
\setstretch{1.5}
\RequirePackage{mathptmx}
\RequirePackage{graphicx}

\renewcommand*\contentsname{\MakeUppercase{Table of Contents}}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\listfigurename}{List of Figures}

\RequirePackage{titlesec}
\titleformat{\chapter}[block]{\normalfont\Large\bfseries}{CHAPTER \thechapter}{0.3em}{\linebreak  \centering\MakeUppercase}
\titlespacing*{\chapter}{0pt}{-1em}{0pt}
\titleformat{\section}[block]{\normalfont\bfseries}{\thesection}{0.3em}{\MakeUppercase}
\titleformat{\subsection}[block]{\normalfont\bfseries}{\thesubsection}{0.3em}{}

\RequirePackage{subcaption}
\RequirePackage{natbib}
\RequirePackage{array}
\RequirePackage[sort]{acro}
\acsetup{
	list/template=tabular
}
\RequirePackage{longtable}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{listings}
\RequirePackage{titlecaps}
\RequirePackage{pdfpages}
\RequirePackage{ragged2e}
\RequirePackage{ifthen}
\RequirePackage{rotating}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{hyperref}
\hypersetup{
	colorlinks=true,
	linkcolor=black,
	citecolor=black,
	urlcolor=blue
}
\lstset{
	captionpos=b,
	frame=tb,
	basicstyle=\scriptsize\ttfamily,
	showstringspaces=false,
	keepspaces=true,
	breaklines=true
}
\bibliographystyle{apalike}
