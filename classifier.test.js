import { trainFromExamples, classifyText, getModel } from './classifier.js'

const examples = [
  { text: 'My phone was stolen from the bus', label: 'fir' },
  { text: 'There was a robbery at my home', label: 'fir' },
  { text: 'Victim of assault near market', label: 'fir' },
  { text: 'Someone threatened me with a knife', label: 'fir' },
  { text: 'Car theft reported last night', label: 'fir' },
  { text: 'Apply for police verification certificate', label: 'non-fir' },
  { text: 'Need address proof and NOC', label: 'non-fir' },
  { text: 'Request clearance certificate for passport', label: 'non-fir' },
  { text: 'Document verification required', label: 'non-fir' },
  { text: 'Service issue with helpdesk', label: 'non-fir' },
  { text: 'Burglary attempt in apartment', label: 'fir' },
  { text: 'Pickpocketing incident reported', label: 'fir' },
  { text: 'Harassment complaint at workplace', label: 'fir' },
  { text: 'Extortion calls received', label: 'fir' },
  { text: 'Application for character certificate', label: 'non-fir' },
  { text: 'Lost document request', label: 'non-fir' },
  { text: 'Passport police verification appointment', label: 'non-fir' },
  { text: 'Attack by unknown persons', label: 'fir' },
  { text: 'Molestation case report', label: 'fir' },
  { text: 'Need NOC for address change', label: 'non-fir' },
  { text: 'I lost my personal item and could not find it despite searching the surrounding area', label: 'non-fir' },
]

trainFromExamples(examples)

let correct = 0
for (const ex of examples) {
  const r = classifyText(ex.text)
  const ok = (r.label.toLowerCase() === ex.label.toLowerCase())
  if (ok) correct += 1
}

const acc = correct / examples.length
console.log('Classifier accuracy', { accuracy: acc.toFixed(3), vocab: getModel().vocab.size })
if (acc < 0.85) {
  console.error('Accuracy below threshold', acc)
  process.exit(1)
}
