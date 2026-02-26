# คำนำ (Preface)

หนังสือเล่มนี้เกิดขึ้นจากคำถามพื้นฐานที่เรียบง่ายแต่ทรงพลัง:

> “เราสามารถอธิบาย YOLO และการตรวจจับวัตถุแบบสมัยใหม่ ด้วยกรอบทฤษฎีที่เป็นหนึ่งเดียวได้หรือไม่?”

ตลอดหลายปีที่ผ่านมา โมเดลตระกูล YOLO พัฒนาอย่างรวดเร็วจากระบบที่อาศัย anchor และ heuristic ไปสู่โครงสร้างที่ซับซ้อนยิ่งขึ้น เช่น decoupled head, dynamic assignment, anchor-free design และแนวคิดแบบ foundation model อย่างไรก็ตาม การอธิบายความก้าวหน้าเหล่านี้มักกระจายอยู่ในเชิงวิศวกรรม มากกว่ากรอบคณิตศาสตร์ที่เป็นระบบ

หนังสือเล่มนี้จึงมีเป้าหมายเพื่อ:

1. วางรากฐาน **Statistical Learning Theory** สำหรับงาน detection
2. เชื่อมโยง **Maximum Likelihood, Empirical Risk Minimization และ Optimization**
3. อธิบาย YOLO ในมุมมองของ **Entropy Reduction และ Information Flow**
4. วิเคราะห์พัฒนาการของ YOLO รุ่นต่าง ๆ ผ่านกรอบ **Bias–Variance–Gradient–Assignment**
5. เสนอคำถามเชิงทฤษฎีที่ยังเปิดอยู่ในยุค Foundation Model

เนื้อหาครอบคลุมตั้งแต่ risk minimization, sample complexity, structured prediction, gradient dynamics, ไปจนถึง information-theoretic limits ของ detection system โดยตั้งใจเขียนให้:

- นักวิจัยระดับบัณฑิตศึกษา
- วิศวกรระบบ deep learning
- ผู้สนใจการออกแบบโมเดล detection อย่างลึกซึ้ง

สามารถมอง YOLO ไม่ใช่เพียง “สถาปัตยกรรมหนึ่ง” แต่เป็น **ระบบการลดความไม่แน่นอนเชิงโครงสร้างของภาพ (structured entropy reduction system)**

หนังสือเล่มนี้ไม่ได้พยายามปิดคำถามทั้งหมด ตรงกันข้าม มันพยายามเปิดคำถามใหม่:

- ขอบเขตล่างเชิง information-theoretic ของ detection คืออะไร?
- ความสัมพันธ์ระหว่าง assignment complexity กับ gradient quality ควรเหมาะสมอย่างไร?
- Flat minima ใน detection สามารถพิสูจน์ได้หรือไม่?
- YOLO สามารถกลายเป็น foundation perception model ได้อย่างไร?

หวังว่าผู้อ่านจะใช้หนังสือเล่มนี้เป็นทั้งแผนที่และเข็มทิศ —
แผนที่เพื่อเข้าใจอดีต
และเข็มทิศเพื่อออกแบบอนาคตของระบบตรวจจับวัตถุ

**Kittawat Thongpud**

19 กุมภาพันธ์ 2026

---
