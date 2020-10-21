# EXERCISES
EXERCISES - proposed project
Project proposed to deeply investigate The Treatment Episode Data Set (TEDS), to understand how demographic
and other factors influence drug/alcohol use and affect user's lives. I would like to see interactions, such 
how variable importance and facor interaction changes with variables such as age, education, etc.

#%% Data explanation
Read here for more info: https://github.com/semework/EXERCISES/blob/master/TEDSA-2017-CODEBOOK.pdf
"""
Center for Behavioral Health Statistics and Quality
Substance Abuse and Mental Health Services Administration

Public Domain Notice
All material appearing in this document is in the public domain and may be reproduced or copied
without permission from SAMHSA.

Recommended Citation
SubstanceAbuse and Mental Health ServicesAdministration, TreatmentEpisode Data Set(TEDS):
2017. Rockville, MD: Substance Abuse and Mental Health Services Administration, 2019.

TREATMENT EPISODE DATA SET — ADMISSIONS
(TEDS-A), 2017

Introduction to TEDS
The Treatment Episode Data Set (TEDS) system serves as a repository of treatment data routinely
collected by states for the purposes of monitoring their substance use treatment systems. It is
comprised of selected data items from states’ administrative records that are converted to a
standardized format which is consistent across allstates. These standardized data constitute TEDS.
The TEDS system is comprised of two major components: the admissions data set (TEDS-A) and
the discharges data set(TEDS-D). Data fortheTEDS-Admissions(TEDS-A)file were firstreported
in 1992, while data for the TEDS-D were first reported in 2000.

Admissions
TEDS-Aprovides demographic, clinical, and substance use characteristics of admissionsto alcohol
or drug treatment in facilities that report to state administrative data systems. The unit of analysis
is treatment admissions to state-licensed or certified substance use treatment centers that receive
federal public funding.
TEDS-A has two parts: a minimum data set and a supplemental data set. The former is collected
by all states; the latter is collected by some.
The minimum data set consists of 19 items that include:
● demographic information

● primary,secondary, and tertiary substances used by the subject, and theirroute of administration,
frequency of use, and age at first use

● source of referral to treatment

● number of prior treatment episodes; and

● service type, including planned use of medication-assisted (i.e., methadone, buprenorphine, or
naltrexone) opioid therapy.

TEDS-A’s supplemental data set includes 15 psychiatric, social, and economic items.

"""

Variable      Source     Type     Length      Label
===================================================
● ADMYR     Computed variable Numeric 8 Year of admission

● AGE       Minimum data set Numeric 8 Age at admission

● ALCDRUG   Computed variable Numeric 8 Substance use type

● ALCFLG    Computed variable Numeric 8 Alcohol reported at admission

● AMPHFLG   Computed variable Numeric 8 Other amphetamines reported at admission

● ARRESTS   Supplemental data set Numeric 8 Number of arrests in the 30 days prior to admission

● BARBFLG   Computed variable Numeric 8 Barbiturates reported at admission

● BENZFLG   Computed variable Numeric 8 Benzodiazepines reported at admission

● CASEID    Computed variable Numeric 8 Case identification number

● CBSA2010  Computed variable Numeric 8 Metropolitan or micropolitan statistical area

● COKEFLG   Computed variable Numeric 8 Cocaine/crack reported at admission

● DAYWAIT   Supplemental data set Numeric 8 Number of days waiting to enter treatment

● DETCRIM   Supplemental data set Numeric 8 Detailed criminal justice referral

● DETNLF    Supplemental data set Numeric 8 Detailed “not in labor force” category at admission

● DIVISION  Computed variable Numeric 8 Census division

● DSMCRIT   Supplemental data set Numeric 8 DSM diagnosis (SuDS 4 or SuDS 19)

● EDUC      Minimum data set Numeric 8 Education

● EMPLOY    Minimum data set Numeric 8 Employment status at admission

● ETHNIC    Minimum data set Numeric 8 Hispanic or Latino origin (ethnicity)

● FREQ1     Minimum data set Numeric 8 Frequency of use at admission (primary substance)

● FREQ2     Minimum data set Numeric 8 Frequency of use at admission (secondary substance)

● FREQ3     Minimum data set Numeric 8 Frequency of use at admission (tertiary substance)

● FREQ_ATND_SELF_HELP    Supplemental data set Numeric 8
                       Frequency of attendance at substance use self-help groups in the 30
                       days prior to admission
                       
● FRSTUSE1  Minimum data set Numeric 8 Age at first use (primary substance)

● FRSTUSE2  Minimum data set Numeric 8 Age at first use (secondary substance)

● FRSTUSE3  Minimum data set Numeric 8 Age at first use (tertiary substance)

● GENDER    Minimum data set Numeric 8 Biologic sex

● HALLFLG   Computed variable Numeric 8 Other hallucinogens reported at admission

● HERFLG    Computed variable Numeric 8 Heroin reported at admission

● HLTHINS   Supplemental data set Numeric 8 Health insurance at admission

● IDU       Computed variable Numeric 8 Current IV drug use reported at admission

● INHFLG    Computed variable Numeric 8 Inhalants reported at admission

● LIVARAG   Supplemental data set Numeric 8 Living arrangements at admission

● MARFLG    Computed variable Numeric 8 Marijuana/hashish reported at admission

● MARSTAT   Supplemental data set Numeric 8 Marital status

● METHFLG   Computed variable Numeric 8 Non-Rx methadone reported at admission

● METHUSE   Minimum data set Numeric 8 Planned medication-assisted opioid therapy

● MTHAMFLG  Computed variable Numeric 8 Methamphetamine reported at admission

● NOPRIOR   Minimum data set Numeric 8 Number of previous substance use treatment episodes

● OPSYNFLG  Computed variable Numeric 8 Other opiates/synthetics reported at admission

● OTCFLG    Computed variable Numeric 8 Over-the-counter medication reported at admission

● OTHERFLG  Computed variable Numeric 8 Other drug reported at admission

● PCPFLG    Computed variable Numeric 8 PCP reported at admission

● PREG      Supplemental data set Numeric 8 Pregnant at admission

● PRIMINC   Supplemental data set Numeric 8 Source of income/support

● PRIMPAY   Supplemental data set Numeric 8 Primary source of payment for treatment

● PSOURCE   Minimum data set Numeric 8 Treatment referral source

● PSYPROB   Supplemental data set Numeric 8 Co-occurring mental and substance use disorders

● RACE      Minimum data set Numeric 8 Race

● REGION    Computed variable Numeric 8 Census region

● ROUTE1    Minimum data set Numeric 8 Usual route of administration (primary substance)

● ROUTE2    Minimum data set Numeric 8 Usual route of administration (secondary substance)

● ROUTE3    Minimum data set Numeric 8 Usual route of administration (tertiary substance)

● SEDHPFLG  Computed variable Numeric 8 Other non-barbiturate sedatives/hypnotics reported at admission

● SERVICES  Minimum data set Numeric 8 Service setting at admission

● STFIPS    Computed variable Numeric 8 Census state FIPS code

● STIMFLG   Computed variable Numeric 8 Other stimulants reported at admission

● SUB1      Minimum data set Numeric 8 Substance use at admission (primary)

● SUB2      Minimum data set Numeric 8 Substance use at admission (secondary)

● SUB3      Minimum data set Numeric 8 Substance use at admission (tertiary)

● TRNQFLG   Computed variable Numeric 8 Other non-benzodiazepine tranquilizers reported at admission

● VET       Supplemental data set Numeric 8 Veteran status
