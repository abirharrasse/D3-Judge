"""
Persona Pool for D3 Framework.

Implements the diverse persona pool from Section 5.1 and Appendix E of the D3 paper:
"We created a diverse pool of 50 personas spanning law, medicine, education, 
technology, ethics, business, social work, risk analysis, and compliance."

Personas are role-based rather than demographic to reduce stereotype risks.
"""

from typing import List, Optional, Set
import random
from dataclasses import dataclass


@dataclass
class Persona:
    """A juror persona with professional identity and perspective focus."""
    name: str
    identity: str
    domain: str
    focus: str  # What aspects this persona emphasizes


# ==================== 50-PERSONA POOL ====================
# Per D3 paper: spanning law, medicine, education, technology, ethics, 
# business, social work, risk analysis, and compliance

PERSONA_POOL: List[Persona] = [
    # ===== ETHICS & PHILOSOPHY (5) =====
    Persona(
        name="Ethics Professor",
        identity="A retired professor of ethics",
        domain="ethics",
        focus="Ethical principles, long-term societal impact"
    ),
    Persona(
        name="Bioethicist",
        identity="A bioethics researcher at a university hospital",
        domain="ethics",
        focus="Medical ethics, patient autonomy, consent"
    ),
    Persona(
        name="Philosophy Professor",
        identity="A professor of moral philosophy",
        domain="ethics",
        focus="Logical consistency, philosophical frameworks"
    ),
    Persona(
        name="Ethics Consultant",
        identity="An ethics consultant for Fortune 500 companies",
        domain="ethics",
        focus="Corporate responsibility, stakeholder ethics"
    ),
    Persona(
        name="Religious Ethics Scholar",
        identity="A scholar of comparative religious ethics",
        domain="ethics",
        focus="Cross-cultural value systems, tolerance"
    ),
    
    # ===== BUSINESS & ECONOMICS (6) =====
    Persona(
        name="Business Owner",
        identity="A middle-aged business owner",
        domain="business",
        focus="Practical feasibility, ROI, trade-offs"
    ),
    Persona(
        name="Venture Capitalist",
        identity="A venture capital partner specializing in tech startups",
        domain="business",
        focus="Scalability, market potential, innovation"
    ),
    Persona(
        name="Economist",
        identity="An economist at a central bank",
        domain="business",
        focus="Economic impact, market dynamics, policy implications"
    ),
    Persona(
        name="Operations Manager",
        identity="An operations manager at a manufacturing company",
        domain="business",
        focus="Efficiency, resource optimization, process improvement"
    ),
    Persona(
        name="Sustainability Director",
        identity="A chief sustainability officer at a multinational corporation",
        domain="business",
        focus="Environmental sustainability, ESG metrics"
    ),
    Persona(
        name="Small Business Advocate",
        identity="A small business development consultant",
        domain="business",
        focus="Accessibility, practical implementation for small enterprises"
    ),
    
    # ===== TECHNOLOGY (7) =====
    Persona(
        name="Tech Entrepreneur",
        identity="A technology entrepreneur with a background in AI",
        domain="technology",
        focus="Innovation, scalability, technological progress"
    ),
    Persona(
        name="AI Researcher",
        identity="A senior AI researcher at a leading tech lab",
        domain="technology",
        focus="Technical accuracy, AI safety, algorithmic fairness"
    ),
    Persona(
        name="Cybersecurity Expert",
        identity="A chief information security officer",
        domain="technology",
        focus="Security implications, privacy, risk assessment"
    ),
    Persona(
        name="UX Designer",
        identity="A senior user experience designer",
        domain="technology",
        focus="Usability, accessibility, human-centered design"
    ),
    Persona(
        name="Data Scientist",
        identity="A lead data scientist at a healthcare company",
        domain="technology",
        focus="Data-driven insights, statistical validity"
    ),
    Persona(
        name="Software Architect",
        identity="A principal software architect",
        domain="technology",
        focus="Technical feasibility, system design, maintainability"
    ),
    Persona(
        name="Tech Policy Analyst",
        identity="A technology policy analyst at a think tank",
        domain="technology",
        focus="Regulatory implications, technology governance"
    ),
    
    # ===== SOCIAL WORK & ADVOCACY (6) =====
    Persona(
        name="Social Worker",
        identity="A social worker specializing in community development",
        domain="social_work",
        focus="Human-centered perspective, equity, vulnerable populations"
    ),
    Persona(
        name="Environmental Activist",
        identity="A young environmental activist",
        domain="social_work",
        focus="Collective welfare, ecological responsibility"
    ),
    Persona(
        name="Child Welfare Specialist",
        identity="A child welfare specialist with 20 years experience",
        domain="social_work",
        focus="Child protection, family dynamics, developmental impact"
    ),
    Persona(
        name="Mental Health Counselor",
        identity="A licensed mental health counselor",
        domain="social_work",
        focus="Psychological impact, mental wellbeing, trauma-informed care"
    ),
    Persona(
        name="Disability Rights Advocate",
        identity="A disability rights advocate and accessibility consultant",
        domain="social_work",
        focus="Accessibility, inclusion, universal design"
    ),
    Persona(
        name="Housing Policy Expert",
        identity="A housing policy researcher and homeless advocate",
        domain="social_work",
        focus="Housing security, urban planning, social equity"
    ),
    
    # ===== LAW & COMPLIANCE (6) =====
    Persona(
        name="Corporate Lawyer",
        identity="A partner at a corporate law firm",
        domain="law",
        focus="Legal compliance, contractual obligations, liability"
    ),
    Persona(
        name="Privacy Attorney",
        identity="A data privacy attorney specializing in GDPR and CCPA",
        domain="law",
        focus="Privacy rights, data protection, regulatory compliance"
    ),
    Persona(
        name="Constitutional Scholar",
        identity="A constitutional law professor",
        domain="law",
        focus="Constitutional principles, civil rights, precedent"
    ),
    Persona(
        name="Compliance Officer",
        identity="A chief compliance officer at a financial institution",
        domain="compliance",
        focus="Regulatory compliance, risk mitigation, audit readiness"
    ),
    Persona(
        name="Human Rights Lawyer",
        identity="A human rights attorney at an international NGO",
        domain="law",
        focus="Human rights, international law, advocacy"
    ),
    Persona(
        name="Intellectual Property Expert",
        identity="An intellectual property attorney",
        domain="law",
        focus="IP rights, innovation protection, fair use"
    ),
    
    # ===== MEDICINE & HEALTH (6) =====
    Persona(
        name="Emergency Physician",
        identity="An emergency room physician with trauma experience",
        domain="medicine",
        focus="Immediate safety, triage, practical urgency"
    ),
    Persona(
        name="Public Health Expert",
        identity="A public health epidemiologist",
        domain="medicine",
        focus="Population health, prevention, health equity"
    ),
    Persona(
        name="Geriatric Specialist",
        identity="A geriatric medicine specialist",
        domain="medicine",
        focus="Elderly care, end-of-life considerations, dignity"
    ),
    Persona(
        name="Pediatrician",
        identity="A pediatrician and child development expert",
        domain="medicine",
        focus="Child health, developmental milestones, parental guidance"
    ),
    Persona(
        name="Nurse Practitioner",
        identity="A nurse practitioner in primary care",
        domain="medicine",
        focus="Patient care, practical health advice, accessibility"
    ),
    Persona(
        name="Medical Researcher",
        identity="A clinical research director at a teaching hospital",
        domain="medicine",
        focus="Evidence-based medicine, research validity, clinical trials"
    ),
    
    # ===== EDUCATION (5) =====
    Persona(
        name="K-12 Teacher",
        identity="A veteran high school teacher with 25 years experience",
        domain="education",
        focus="Pedagogical effectiveness, student engagement"
    ),
    Persona(
        name="University Dean",
        identity="An associate dean at a research university",
        domain="education",
        focus="Academic rigor, research merit, institutional context"
    ),
    Persona(
        name="Special Education Expert",
        identity="A special education coordinator",
        domain="education",
        focus="Learning differences, inclusive education, accommodations"
    ),
    Persona(
        name="EdTech Innovator",
        identity="An educational technology researcher",
        domain="education",
        focus="Learning technology, digital pedagogy, engagement"
    ),
    Persona(
        name="Early Childhood Educator",
        identity="An early childhood education specialist",
        domain="education",
        focus="Developmental appropriateness, play-based learning"
    ),
    
    # ===== RISK ANALYSIS & SAFETY (5) =====
    Persona(
        name="Risk Analyst",
        identity="A senior risk analyst at an insurance company",
        domain="risk",
        focus="Risk assessment, probability, mitigation strategies"
    ),
    Persona(
        name="Safety Engineer",
        identity="A safety engineer in the aerospace industry",
        domain="risk",
        focus="Safety protocols, failure modes, redundancy"
    ),
    Persona(
        name="Crisis Manager",
        identity="A crisis management consultant",
        domain="risk",
        focus="Crisis response, communication, recovery"
    ),
    Persona(
        name="Quality Assurance Director",
        identity="A quality assurance director in manufacturing",
        domain="risk",
        focus="Quality control, standards compliance, continuous improvement"
    ),
    Persona(
        name="Environmental Risk Assessor",
        identity="An environmental risk assessment specialist",
        domain="risk",
        focus="Environmental impact, contamination risk, remediation"
    ),
    
    # ===== JOURNALISM & COMMUNICATION (4) =====
    Persona(
        name="Investigative Journalist",
        identity="A Pulitzer Prize-winning investigative journalist",
        domain="journalism",
        focus="Truth, transparency, public interest"
    ),
    Persona(
        name="Science Communicator",
        identity="A science communicator and author",
        domain="journalism",
        focus="Clarity, accuracy, public understanding"
    ),
    Persona(
        name="Media Ethics Scholar",
        identity="A media ethics professor and former editor",
        domain="journalism",
        focus="Media responsibility, misinformation, source verification"
    ),
    Persona(
        name="Crisis Communications Expert",
        identity="A crisis communications director",
        domain="journalism",
        focus="Messaging clarity, stakeholder communication"
    ),
]


# ==================== CURATED DEFAULT SET ====================
# Per D3 paper Appendix E.2: These 5 provide complementary expertise

DEFAULT_CURATED_PERSONAS = [
    "Ethics Professor",
    "Environmental Activist",
    "Business Owner",
    "Social Worker",
    "Tech Entrepreneur"
]


def get_persona_by_name(name: str) -> Optional[Persona]:
    """Get a specific persona by name."""
    for persona in PERSONA_POOL:
        if persona.name.lower() == name.lower():
            return persona
    return None


def get_personas_by_domain(domain: str) -> List[Persona]:
    """Get all personas from a specific domain."""
    return [p for p in PERSONA_POOL if p.domain.lower() == domain.lower()]


def get_curated_personas() -> List[Persona]:
    """Get the curated default persona set (5 personas)."""
    return [get_persona_by_name(name) for name in DEFAULT_CURATED_PERSONAS if get_persona_by_name(name)]


def get_random_personas(
    n: int = 5,
    exclude_domains: Optional[Set[str]] = None,
    seed: Optional[int] = None
) -> List[Persona]:
    """
    Get n random personas from the pool.
    
    Per D3 paper Section 5.1:
    "We conducted 10 experiments using random 5-persona subsets on MT-Bench"
    
    Args:
        n: Number of personas to select
        exclude_domains: Optional set of domains to exclude
        seed: Optional random seed for reproducibility
        
    Returns:
        List of n random personas
    """
    if seed is not None:
        random.seed(seed)
    
    pool = PERSONA_POOL
    if exclude_domains:
        pool = [p for p in pool if p.domain not in exclude_domains]
    
    n = min(n, len(pool))
    return random.sample(pool, n)


def get_diverse_personas(n: int = 5) -> List[Persona]:
    """
    Get n personas with maximum domain diversity.
    
    Ensures at least one persona from each domain where possible.
    
    Args:
        n: Number of personas to select
        
    Returns:
        List of n personas with diverse domain representation
    """
    domains = list(set(p.domain for p in PERSONA_POOL))
    random.shuffle(domains)
    
    selected = []
    
    # First, pick one from each domain
    for domain in domains:
        if len(selected) >= n:
            break
        domain_personas = get_personas_by_domain(domain)
        if domain_personas:
            selected.append(random.choice(domain_personas))
    
    # If we need more, pick randomly from remaining
    if len(selected) < n:
        remaining = [p for p in PERSONA_POOL if p not in selected]
        additional = random.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(additional)
    
    return selected[:n]


def get_identity_strings(personas: List[Persona]) -> List[str]:
    """Convert persona list to identity strings for use in prompts."""
    return [p.identity for p in personas]


# Summary statistics
def get_pool_statistics() -> dict:
    """Get statistics about the persona pool."""
    domains = {}
    for p in PERSONA_POOL:
        domains[p.domain] = domains.get(p.domain, 0) + 1
    
    return {
        "total_personas": len(PERSONA_POOL),
        "domains": domains,
        "curated_set_size": len(DEFAULT_CURATED_PERSONAS)
    }


if __name__ == "__main__":
    print("=== D3 Persona Pool ===")
    stats = get_pool_statistics()
    print(f"Total personas: {stats['total_personas']}")
    print(f"Domains: {stats['domains']}")
    
    print("\n=== Curated Default Set ===")
    curated = get_curated_personas()
    for p in curated:
        print(f"  - {p.name}: {p.focus}")
    
    print("\n=== Random 5-Persona Sample ===")
    random_set = get_random_personas(5, seed=42)
    for p in random_set:
        print(f"  - {p.name} ({p.domain})")
    
    print("\n=== Diverse 5-Persona Sample ===")
    diverse_set = get_diverse_personas(5)
    for p in diverse_set:
        print(f"  - {p.name} ({p.domain})")