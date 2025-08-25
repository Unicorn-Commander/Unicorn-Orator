// Voice descriptions and quality ratings for Unicorn Orator
const voiceData = {
  // American Female voices
  "af": { 
    name: "AF - Default", 
    quality: 4, 
    description: "Standard American female voice, clear and professional",
    gender: "female",
    accent: "American"
  },
  "af_alloy": { 
    name: "AF Alloy", 
    quality: 5, 
    description: "Smooth, metallic undertone, excellent for technical content",
    gender: "female",
    accent: "American"
  },
  "af_aoede": { 
    name: "AF Aoede", 
    quality: 4, 
    description: "Musical quality, named after the Greek muse of song",
    gender: "female",
    accent: "American"
  },
  "af_bella": { 
    name: "AF Bella", 
    quality: 5, 
    description: "Warm and friendly, perfect for conversational content",
    gender: "female",
    accent: "American"
  },
  "af_heart": { 
    name: "AF Heart", 
    quality: 5, 
    description: "Emotional and expressive, great for storytelling",
    gender: "female",
    accent: "American"
  },
  "af_jessica": { 
    name: "AF Jessica", 
    quality: 4, 
    description: "Professional newsreader style, clear articulation",
    gender: "female",
    accent: "American"
  },
  "af_kore": { 
    name: "AF Kore", 
    quality: 4, 
    description: "Youthful and energetic voice",
    gender: "female",
    accent: "American"
  },
  "af_nicole": { 
    name: "AF Nicole", 
    quality: 4, 
    description: "Sophisticated and mature tone",
    gender: "female",
    accent: "American"
  },
  "af_nova": { 
    name: "AF Nova", 
    quality: 5, 
    description: "Modern, bright voice with excellent clarity",
    gender: "female",
    accent: "American"
  },
  "af_river": { 
    name: "AF River", 
    quality: 4, 
    description: "Flowing, calm voice ideal for meditation content",
    gender: "female",
    accent: "American"
  },
  "af_sarah": { 
    name: "AF Sarah", 
    quality: 4, 
    description: "Friendly neighborhood voice, approachable",
    gender: "female",
    accent: "American"
  },
  "af_sky": { 
    name: "AF Sky", 
    quality: 5, 
    description: "Light, airy voice with excellent range",
    gender: "female",
    accent: "American"
  },
  
  // American Male voices
  "am_adam": { 
    name: "AM Adam", 
    quality: 4, 
    description: "Deep, authoritative male voice",
    gender: "male",
    accent: "American"
  },
  "am_echo": { 
    name: "AM Echo", 
    quality: 5, 
    description: "Resonant voice with natural reverb quality",
    gender: "male",
    accent: "American"
  },
  "am_eric": { 
    name: "AM Eric", 
    quality: 4, 
    description: "Casual, friendly male voice",
    gender: "male",
    accent: "American"
  },
  "am_fenrir": { 
    name: "AM Fenrir", 
    quality: 5, 
    description: "Powerful, commanding voice named after Norse wolf",
    gender: "male",
    accent: "American"
  },
  "am_liam": { 
    name: "AM Liam", 
    quality: 4, 
    description: "Young adult male, contemporary sound",
    gender: "male",
    accent: "American"
  },
  "am_michael": { 
    name: "AM Michael", 
    quality: 4, 
    description: "Classic male narrator voice",
    gender: "male",
    accent: "American"
  },
  "am_onyx": { 
    name: "AM Onyx", 
    quality: 5, 
    description: "Deep, rich voice with gravitas",
    gender: "male",
    accent: "American"
  },
  "am_puck": { 
    name: "AM Puck", 
    quality: 4, 
    description: "Playful, mischievous character voice",
    gender: "male",
    accent: "American"
  },
  "am_santa": { 
    name: "AM Santa", 
    quality: 4, 
    description: "Jolly, warm voice perfect for holiday content",
    gender: "male",
    accent: "American"
  },
  
  // British Female voices
  "bf_emma": { 
    name: "BF Emma", 
    quality: 5, 
    description: "Elegant British female voice, BBC quality",
    gender: "female",
    accent: "British"
  },
  "bf_isabella": { 
    name: "BF Isabella", 
    quality: 5, 
    description: "Sophisticated British accent, perfect for audiobooks",
    gender: "female",
    accent: "British"
  },
  
  // British Male voices
  "bm_george": { 
    name: "BM George", 
    quality: 5, 
    description: "Distinguished British gentleman voice",
    gender: "male",
    accent: "British"
  },
  "bm_lewis": { 
    name: "BM Lewis", 
    quality: 4, 
    description: "Modern British male voice, clear pronunciation",
    gender: "male",
    accent: "British"
  }
};

// Helper function to get quality stars
function getQualityStars(rating) {
  return '★'.repeat(rating) + '☆'.repeat(5 - rating);
}

// Helper function to group voices by category
function groupVoices() {
  const groups = {
    'American Female': [],
    'American Male': [],
    'British Female': [],
    'British Male': [],
    'Other': []
  };
  
  Object.entries(voiceData).forEach(([id, data]) => {
    const category = `${data.accent} ${data.gender.charAt(0).toUpperCase() + data.gender.slice(1)}`;
    if (groups[category]) {
      groups[category].push({ id, ...data });
    } else {
      groups['Other'].push({ id, ...data });
    }
  });
  
  return groups;
}